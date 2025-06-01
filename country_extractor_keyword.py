# requirements: prefect>=2.0, dateparser, pycountry, countryguess
# pip install prefect dateparser pycountry countryguess rich

from prefect import task, flow
import logging
import json
import time
import csv
from datetime import datetime
from typing import List, Dict
import pycountry
from rich.console import Console
from pathlib import Path
from collections import Counter
import re

# Create a logger
logger = logging.getLogger('country_extractor')
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create a file handler
log_file = log_dir / f"country_extraction_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
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

@task
def extract_countries_keyword(text: str) -> List[str]:
    """
    Extract country names from text using keyword matching with pycountry.
    Returns a list of unique country names found.
    """
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    found_countries = set()
    
    # Try to find exact matches from our valid countries set
    for country in pycountry.countries:
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(country.name.lower()) + r'\b'
        if re.search(pattern, text_lower):
            # Convert back to proper case using pycountry
            found_countries.add(country.name)
    
    return list(found_countries)

@task
def process_document(document: dict) -> dict:
    """
    Process a single document and extract countries from each page.
    Returns a dictionary with document info and country frequencies.
    """
    doc_name = document["document_name"]
    page_texts = document["page_texts"]
    
    # Store results for each page
    page_results = {}
    all_countries = Counter()
    
    for page_num, text in page_texts.items():
        # Extract countries from this page
        countries = extract_countries_keyword(text)
        
        # Count frequencies for this page
        page_countries = Counter(countries)
        
        # Update overall document counter
        all_countries.update(countries)
        
        # Store results for this page
        page_results[page_num] = {
            "countries": dict(page_countries),
            "total_countries": len(countries)
        }
    
    return {
        "document_name": doc_name,
        "total_pages": len(page_texts),
        "total_countries_found": len(all_countries),
        "countries_by_page": page_results,
        "overall_country_frequencies": dict(all_countries),
        "extraction_method": "keyword"
    }

@task
def save_to_csv(results: List[dict], output_file: Path):
    """
    Save country analysis results to a CSV file with columns:
    document_name, page_number, country, frequency
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['document_name', 'page_number', 'country', 'frequency'])
            
            # Write data rows
            for doc in results:
                doc_name = doc['document_name']
                for page_num, page_data in doc['countries_by_page'].items():
                    for country, freq in page_data['countries'].items():
                        writer.writerow([doc_name, page_num, country, freq])
        
        logger.info(f"CSV results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving CSV file: {e}", exc_info=True)
        raise

@flow
def process_extracted_data():
    """Main flow to process the extracted JSON data and identify countries using keyword-based extraction."""
    console = Console()
    logger.info("Starting country extraction pipeline using keyword-based method")
    console.print("\n[bold]Starting keyword-based country extraction[/bold]")
    
    input_file = Path("output/extracted_data.json")
    output_json = Path("output/country_analysis_keyword.json")
    output_csv = Path("output/country_analysis_keyword.csv")
    
    if not input_file.exists():
        error_msg = f"Input file not found: {input_file}"
        logger.error(error_msg)
        console.print(f"\n[red]{error_msg}[/red]")
        return
    
    try:
        # Load the extracted data
        with open(input_file, 'r', encoding='utf-8') as f:
            documents = json.load(f)
        
        total_docs = len(documents)
        logger.info(f"Processing {total_docs} documents")
        console.print(f"\n[bold]Processing {total_docs} documents...[/bold]")
        
        # Process each document
        results = []
        with console.status("[bold green]Processing documents...") as status:
            for i, doc in enumerate(documents, 1):
                status.update(f"[bold green]Processing document {i}/{total_docs}: {doc['document_name']}")
                logger.info(f"Processing document {i}/{total_docs}: {doc['document_name']}")
                result = process_document(doc)
                results.append(result)
                logger.info(f"Found {result['total_countries_found']} unique countries in {doc['document_name']}")
        
        # Save JSON results
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save CSV results
        save_to_csv(results, output_csv)
        
        success_msg = f"Analysis complete. Results saved to:\n- {output_json}\n- {output_csv}"
        logger.info(success_msg)
        console.print(f"\n[green]{success_msg}[/green]")
        
    except Exception as e:
        error_msg = f"Error processing data: {e}"
        logger.error(error_msg, exc_info=True)
        console.print(f"\n[red]{error_msg}[/red]")

if __name__ == "__main__":
    start_time = time.time()
    logger.info("Starting country extraction process")
    process_extracted_data()
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds") 