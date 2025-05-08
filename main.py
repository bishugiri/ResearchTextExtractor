# requirements: prefect>=2.0, dateparser, pycountry
# pip install prefect dateparser pycountry

from prefect import task, flow
import re
import dateparser
import pycountry
import pandas as pd
import os
import fitz  # PyMuPDF
from langdetect import detect
import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple
from countryguess import guess_country  # Changed from CountryGuess

# Create a logger
logger = logging.getLogger('pdf_extractor')
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create a file handler
log_file = os.path.join(log_dir, f"pdf_extraction_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter - removed seconds from the timestamp format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                            datefmt='%Y-%m-%d %H:%M')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def log_separator():
    """Add a visual separator in the log file"""
    separator = "-" * 80
    logger.info(separator)

# ----- helpers -------------------------------------------------------------

# Prepare a mapping from country name to ISO3 code
COUNTRY_NAME_TO_ISO3 = {c.name: c.alpha_3 for c in pycountry.countries}
COUNTRY_SET = set(COUNTRY_NAME_TO_ISO3.keys())

DATE_REGEX = re.compile(
    r"\b(?:\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{2,4})\b",
    re.IGNORECASE,
)

# ----- Prefect tasks -------------------------------------------------------

@task(retries=2, retry_delay_seconds=5)
def extract_countries(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z]+", text)
    countries = {w.title() for w in words if w.title() in COUNTRY_SET}
    return sorted(countries)

@task(retries=2, retry_delay_seconds=5)
def extract_dates(text: str) -> list[str]:
    raw_dates = DATE_REGEX.findall(text)
    parsed = set()
    for token in raw_dates:
        dt = dateparser.parse(token, settings={"DATE_ORDER": "DMY"})
        if dt:
            parsed.add(dt.date().isoformat())
    return sorted(parsed)

@task(retries=2, retry_delay_seconds=5)
def extract_iso3_codes(text: str):
    words = re.findall(r"[A-Za-z]+", text)
    found = set()
    for w in words:
        country = w.title()
        if country in COUNTRY_SET:
            found.add(country)
    iso3_codes = sorted([COUNTRY_NAME_TO_ISO3[country] for country in found])
    return iso3_codes

# ----- Flow ----------------------------------------------------------------

@flow(name="text-entity-extractor")
def parse_text_flow(text: str):
    countries = extract_countries(text)
    dates = extract_dates(text)
    return {"countries": countries, "dates": dates}

@flow(name="country-iso3-extractor")
def country_iso3_flow(text: str):
    return extract_iso3_codes(text)

@flow(name="document-iso3-extractor")
def document_iso3_flow(document_id: str, text: str):
    iso3_codes = extract_iso3_codes(text)
    return {
        "document_id": document_id,
        "iso3_codes": iso3_codes
    }

# ----- batch processing from parquet -----

@task
def extract_text_from_pdf(pdf_path):
    logger.info(f"Starting extraction for {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        if not text:
            logger.warning(f"No text found in {pdf_path}")
            return None, None
        logger.info(f"Extraction completed for {pdf_path}")
        return text, detect(text)
    except Exception as e:
        logger.error(f"Error extracting {pdf_path}: {e}")
        return None, None

@task
def process_pdf(pdf_path):
    text, detected_language = extract_text_from_pdf(pdf_path)
    if text:
        word_count = len(text.split())
        if word_count < 50:
            logger.warning(f"Skipping {os.path.basename(pdf_path)}: less than 50 words ({word_count})")
            return None
        
        # Get countries with spell checking and frequencies
        country_dict = get_country_keywords()
        countries = find_countries_with_spellcheck(text, country_dict)
        
        # Calculate total country mentions
        total_country_mentions = sum(country['frequency'] for country in countries)
        
        metadata = {
            "total_characters": len(text),
            "total_words": word_count,
            "detected_language": detected_language,
            "countries": countries,  # Now includes frequencies
            "total_country_mentions": total_country_mentions
        }
        
        # Log country frequencies
        logger.info(f"Country frequencies in {os.path.basename(pdf_path)}:")
        for country in countries:
            logger.info(f"  {country['name']}: {country['frequency']} mentions")
            if country['method'] == 'spellcheck':
                logger.info(f"    (includes spell-checked matches with confidence: {country['confidence']:.2f})")
        
        return {
            "document_name": os.path.basename(pdf_path),
            "text": text,
            "metadata": metadata
        }
    return None

def get_country_keywords() -> Dict[str, str]:
    """
    Create a comprehensive dictionary of country names and their ISO codes.
    Includes official names, common names, and alternative names.
    """
    country_dict = {}
    
    for country in pycountry.countries:
        # Add official name
        country_dict[country.name.lower()] = country.alpha_3
        
        # Add common name if different
        if hasattr(country, 'common_name'):
            country_dict[country.common_name.lower()] = country.alpha_3
            
        # Add alternative names
        if hasattr(country, 'alt_spellings'):
            for alt_name in country.alt_spellings:
                country_dict[alt_name.lower()] = country.alpha_3
                
        # Add country codes as keywords
        country_dict[country.alpha_2.lower()] = country.alpha_3
        country_dict[country.alpha_3.lower()] = country.alpha_3
        
        # Add numeric code
        if hasattr(country, 'numeric'):
            country_dict[country.numeric] = country.alpha_3
    
    return country_dict

def find_countries_with_spellcheck(text: str, country_dict: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Find countries in text using both keyword search and countryguess for spell checking.
    Now includes frequency count for each country.
    
    Args:
        text (str): Text to search for countries
        country_dict (Dict[str, str]): Dictionary of country names and their ISO codes
        
    Returns:
        List[Dict[str, str]]: List of dictionaries containing country names, ISO codes, confidence scores, and frequencies
    """
    found_countries = []
    text_lower = text.lower()
    
    # Track frequencies for each country
    country_frequencies = {}
    
    # First pass: Direct keyword matching
    for country_name, iso_code in country_dict.items():
        pattern = r'\b' + re.escape(country_name) + r'\b'
        matches = re.findall(pattern, text_lower)
        if matches:
            country = pycountry.countries.get(alpha_3=iso_code)
            if country:
                frequency = len(matches)
                country_frequencies[iso_code] = frequency
                found_countries.append({
                    'name': country.name,
                    'code': iso_code,
                    'confidence': 1.0,  # Direct match
                    'method': 'exact',
                    'frequency': frequency
                })
    
    # Second pass: Spell checking with countryguess
    words = re.findall(r"[A-Za-z]+", text)
    spellcheck_frequencies = {}
    
    for word in words:
        if len(word) > 3:  # Only check words longer than 3 characters
            result = guess_country(word)
            if result and result.get('confidence', 0) > 0.5:
                country_code = result.get('country')
                if country_code:
                    country = pycountry.countries.get(alpha_3=country_code)
                    if country:
                        # Update frequency for spell-checked matches
                        if country_code not in spellcheck_frequencies:
                            spellcheck_frequencies[country_code] = 0
                        spellcheck_frequencies[country_code] += 1
                        
                        # Only add if not already found in exact matches
                        if country_code not in country_frequencies:
                            found_countries.append({
                                'name': country.name,
                                'code': country_code,
                                'confidence': result.get('confidence', 0),
                                'method': 'spellcheck',
                                'original_word': word,
                                'frequency': spellcheck_frequencies[country_code]
                            })
    
    # Remove duplicates while preserving order and combining frequencies
    seen = set()
    unique_countries = []
    for country in found_countries:
        if country['code'] not in seen:
            seen.add(country['code'])
            # If country was found by both methods, combine frequencies
            if country['code'] in country_frequencies and country['code'] in spellcheck_frequencies:
                country['frequency'] = country_frequencies[country['code']] + spellcheck_frequencies[country['code']]
            unique_countries.append(country)
    
    # Sort countries by frequency (descending)
    unique_countries.sort(key=lambda x: x['frequency'], reverse=True)
    
    return unique_countries

def create_country_table(text_files: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    Create a table of documents with their countries as lists.
    
    Args:
        text_files (List[Tuple[str, str]]): List of (document_id, file_path) tuples
        
    Returns:
        pd.DataFrame: Table with document IDs and country lists
    """
    results = []
    
    for doc_id, file_path in text_files:
        try:
            # Read the text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Find countries in the text
            country_dict = get_country_keywords()
            countries = find_countries_with_spellcheck(text, country_dict)
            
            # Create country lists
            country_names = [country['name'] for country in countries]
            country_codes = [country['code'] for country in countries]
            
            # Add to results
            results.append({
                'document_id': doc_id,
                'countries': ', '.join(country_names),
                'iso_codes': ', '.join(country_codes)
            })
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    return df

def save_country_table(df: pd.DataFrame, output_file: str = "country_table.csv"):
    """
    Save the country table to a CSV file.
    
    Args:
        df (pd.DataFrame): Country table
        output_file (str): Path to save the CSV file
    """
    df.to_csv(output_file, index=False)
    logger.info(f"Country table saved to {output_file}")

@flow
def process_pdfs_in_folder():
    logger.info("Starting PDF processing workflow")
    input_folder = "input"
    output_folder = "output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    records = []
    for i, pdf_file in enumerate(pdf_files, 1):
        if i > 1:  # Add separator before each PDF except the first one
            log_separator()
            
        pdf_path = os.path.join(input_folder, pdf_file)
        logger.info(f"Processing PDF {i}/{len(pdf_files)}: {pdf_file}")
        record = process_pdf(pdf_path)
        if record:
            records.append(record)
            logger.info(f"Successfully processed {pdf_file}")
    
    if records:
        log_separator()  # Add separator before saving results
        # Save to Parquet
        parquet_path = os.path.join(output_folder, "extracted_data.parquet")
        df = pd.DataFrame(records)
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Saved {len(records)} records to {parquet_path}")

        # Save to JSON
        json_path = os.path.join(output_folder, "extracted_data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        logger.info(f"Saved {len(records)} records to {json_path}")

        # Create and save country table
        text_files = [(i+1, os.path.join(output_folder, f"{record['document_name']}.txt")) 
                     for i, record in enumerate(records)]
        country_df = create_country_table(text_files)
        save_country_table(country_df, os.path.join(output_folder, "country_table.csv"))
    else:
        logger.warning("No PDF files processed or no text extracted.")

def process_parquet_file():
    """
    Process the Parquet file and extract countries from the text column.
    """
    logger.info("Starting country extraction from Parquet file")
    
    # Read the Parquet file
    parquet_path = os.path.join("output", "extracted_data.parquet")
    if not os.path.exists(parquet_path):
        logger.error(f"Parquet file not found at {parquet_path}")
        return
    
    df = pd.read_parquet(parquet_path)
    logger.info(f"Found {len(df)} documents to process")
    
    # Get country dictionary
    country_dict = get_country_keywords()
    logger.info(f"Loaded {len(country_dict)} country keywords")
    
    # Process each document
    results = []
    for idx, row in df.iterrows():
        log_separator()
        logger.info(f"Processing document {idx + 1}/{len(df)}: {row['document_name']}")
        
        # Find countries in text
        countries = find_countries_with_spellcheck(row['text'], country_dict)
        
        # Sort countries by frequency (highest to lowest)
        countries.sort(key=lambda x: x['frequency'], reverse=True)
        
        # Create a list of country codes with their frequencies
        country_frequencies = [f"{country['code']}:{country['frequency']}" for country in countries]
        
        # Add to results
        results.append({
            'document_name': row['document_name'],
            'country_frequencies': country_frequencies,
            'total_countries': len(countries),
            'total_mentions': sum(country['frequency'] for country in countries)
        })
        
        # Log detailed progress
        logger.info(f"Found {len(countries)} countries in {row['document_name']}")
        logger.info("Country frequencies (sorted by frequency):")
        for country in countries:
            logger.info(f"  {country['code']}: {country['frequency']} mentions")
            if country['method'] == 'spellcheck':
                logger.info(f"    (includes spell-checked matches with confidence: {country['confidence']:.2f})")
    
    # Create and save results
    if results:
        # Save to CSV
        output_df = pd.DataFrame(results)
        csv_path = os.path.join("output", "country_extraction_results.csv")
        output_df.to_csv(csv_path, index=False)
        logger.info(f"Saved results to {csv_path}")
        
        # Save to JSON
        json_path = os.path.join("output", "country_extraction_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {json_path}")
        
        # Create a detailed log file
        log_path = os.path.join("logs", f"country_extraction_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Country Extraction Results\n")
            f.write("=" * 50 + "\n\n")
            
            for result in results:
                f.write(f"Document: {result['document_name']}\n")
                f.write("-" * 50 + "\n")
                f.write(f"Total unique countries: {result['total_countries']}\n")
                f.write(f"Total country mentions: {result['total_mentions']}\n")
                f.write("\nCountry frequencies:\n")
                for country_freq in result['country_frequencies']:
                    code, freq = country_freq.split(':')
                    f.write(f"  {code}: {freq} mentions\n")
                f.write("\n")
            
            f.write("\nEnd of Report\n")
            f.write("=" * 50 + "\n")
        
        logger.info(f"Created detailed log file at {log_path}")
    else:
        logger.warning("No countries found in any documents")

if __name__ == "__main__":
    try:
        logger.info("Starting country extraction process")
        start_time = time.time()
        process_parquet_file()
        end_time = time.time()
        log_separator()
        logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
