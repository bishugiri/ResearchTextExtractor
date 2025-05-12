# requirements: prefect>=2.0, dateparser, pycountry
# pip install prefect dateparser pycountry countryguess gliner torch rich

from prefect import task, flow
import pandas as pd
import os
import logging
import json
import time
from datetime import datetime
from typing import List, Dict
from countryguess import guess_country
import pycountry
from gliner import GLiNER
from rich.console import Console

# Create a logger
logger = logging.getLogger('country_extractor')
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create a file handler
log_file = os.path.join(log_dir, f"country_extraction_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter
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

def find_countries_keyword_based(text: str) -> List[Dict[str, str]]:
    """
    Find countries in text using keyword-based search against pycountry's country list.
    Returns a list of found countries with their ISO3 codes.
    """
    found_countries = []
    words = text.split()  # Split text into words
    
    # Track frequencies for each country
    country_frequencies = {}
    
    # Get all country names and their ISO3 codes
    country_names = {}
    for country in pycountry.countries:
        country_names[country.name.lower()] = country.alpha_3
        if hasattr(country, 'common_name'):
            country_names[country.common_name.lower()] = country.alpha_3
    
    for word in words:
        word_lower = word.lower()
        if word_lower in country_names:
            iso3_code = country_names[word_lower]
            if iso3_code not in country_frequencies:
                country_frequencies[iso3_code] = 0
            country_frequencies[iso3_code] += 1
    
    # Convert frequencies to list of country information
    for iso3_code, frequency in country_frequencies.items():
        found_countries.append({
            'code': iso3_code,
            'frequency': frequency
        })
    
    # Sort countries by frequency (highest to lowest)
    found_countries.sort(key=lambda x: x['frequency'], reverse=True)
    
    return found_countries

def find_countries_countryguess(text: str) -> List[Dict[str, str]]:
    """
    Find countries in text using countryguess package.
    Returns a list of found countries with their ISO3 codes.
    """
    found_countries = []
    words = text.split()  # Split text into words
    
    # Track frequencies for each country
    country_frequencies = {}
    
    for word in words:
        # Try to guess country from the word
        result = guess_country(word)
        if result and result.get('iso3'):
            iso3_code = result['iso3']
            if iso3_code not in country_frequencies:
                country_frequencies[iso3_code] = 0
            country_frequencies[iso3_code] += 1
    
    # Convert frequencies to list of country information
    for iso3_code, frequency in country_frequencies.items():
        found_countries.append({
            'code': iso3_code,
            'frequency': frequency
        })
    
    # Sort countries by frequency (highest to lowest)
    found_countries.sort(key=lambda x: x['frequency'], reverse=True)
    
    return found_countries

def find_countries_gliner_based(text: str, model: GLiNER) -> List[Dict[str, str]]:
    """
    Find countries in text using GLiNER model to identify geographical entities.
    Returns a list of found countries with their ISO3 codes.
    """
    found_countries = []
    country_frequencies = {}
    
    # Use GLiNER to predict entities
    entities = model.predict_entities(text, ["countries"])
    
    # Process each entity found
    for entity in entities:
        if entity['label'] == 'countries':
            # Try to get ISO3 code using countryguess
            result = guess_country(entity['text'])
            if result and result.get('iso3'):
                iso3_code = result['iso3']
                if iso3_code not in country_frequencies:
                    country_frequencies[iso3_code] = 0
                country_frequencies[iso3_code] += 1
    
    # Convert frequencies to list of country information
    for iso3_code, frequency in country_frequencies.items():
        found_countries.append({
            'code': iso3_code,
            'frequency': frequency
        })
    
    # Sort countries by frequency (highest to lowest)
    found_countries.sort(key=lambda x: x['frequency'], reverse=True)
    
    return found_countries

@flow
def process_json_file(algorithm: str = "countryguess"):
    """
    Process the JSON file and extract countries from the text column.
    Only processes documents with language "en".
    
    Args:
        algorithm (str): The algorithm to use for country extraction.
            Options: "keyword", "countryguess", or "gliner" (default: "countryguess")
    """
    logger.info(f"Starting country extraction from JSON file using {algorithm} algorithm")
    
    # Select the appropriate algorithm
    if algorithm.lower() == "keyword":
        find_countries_func = find_countries_keyword_based
    elif algorithm.lower() == "countryguess":
        find_countries_func = find_countries_countryguess
    elif algorithm.lower() == "gliner":
        # Load GLiNER model
        try:
            model = GLiNER.from_pretrained("urchade/gliner_base")
            find_countries_func = lambda text: find_countries_gliner_based(text, model)
            logger.info("Successfully loaded GLiNER model")
        except Exception as e:
            logger.error(f"Error loading GLiNER model: {str(e)}")
            return
    else:
        logger.error(f"Invalid algorithm specified: {algorithm}. Using countryguess as default.")
        find_countries_func = find_countries_countryguess
    
    # Read the JSON file with proper encoding
    json_path = os.path.join("output", "extracted_data.json")
    if not os.path.exists(json_path):
        logger.error(f"JSON file not found at {json_path}")
        return
    
    # Read JSON file with proper encoding
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    logger.info(f"Found {len(df)} total documents")
    
    # Extract language from metadata and filter for English documents
    df['language'] = df['metadata'].apply(lambda x: x.get('detected_language', '') if isinstance(x, dict) else '')
    df_en = df[df['language'] == 'en']
    logger.info(f"Found {len(df_en)} English documents to process")
    
    # Process each document
    results = []
    for idx, row in df_en.iterrows():
        log_separator()
        logger.info(f"Processing document {idx + 1}/{len(df_en)}: {row['document_name']}")
        
        # Find countries in text using selected algorithm
        countries = find_countries_func(row['text'])
        
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
    
    # Create and save results
    if results:
        # Save to CSV with proper encoding
        output_df = pd.DataFrame(results)
        csv_path = os.path.join("output", f"country_extraction_results_{algorithm}.csv")
        output_df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"Saved results to {csv_path}")
        
        # Save to JSON with proper encoding and ensure_ascii=False
        json_path = os.path.join("output", f"country_extraction_results_{algorithm}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved results to {json_path}")
        
        # Create a detailed log file with proper encoding
        log_path = os.path.join("logs", f"country_extraction_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"Country Extraction Results (English Documents Only) - Using {algorithm} algorithm\n")
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
        logger.warning("No countries found in any English documents")

if __name__ == "__main__":
    start_time = time.time()
    logger.info("Starting country extraction process")
    # You can change the algorithm here: "keyword", "countryguess", or "gliner"
    process_json_file(algorithm="gliner")
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds") 