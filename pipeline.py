# requirements: prefect>=2.0, dateparser, pycountry, countryguess, gliner, transformers, torch
# pip install prefect dateparser pycountry countryguess gliner transformers torch rich

from prefect import task, flow
import pandas as pd
import os
import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Literal
from countryguess import guess_country
import pycountry
from rich.console import Console
from rich.prompt import Prompt
from pathlib import Path
from collections import Counter
import re
# Remove GLiNER imports from top level
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# from gliner import GLiNER
# import torch

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

def get_extraction_method() -> str:
    """
    Get user input for selecting the extraction method.
    Returns either 'keyword' or 'gliner'.
    """
    console = Console()
    
    console.print("\n[bold blue]Select Extraction Method:[/bold blue]")
    console.print("1. Keyword-based extraction (faster, uses exact matching)")
    console.print("2. GLiNER-based extraction (more accurate, uses NLP model)")
    
    while True:
        choice = Prompt.ask(
            "\nEnter your choice (1 or 2)",
            choices=["1", "2"],
            default="1"
        )
        
        if choice == "1":
            console.print("\n[green]Selected: Keyword-based extraction[/green]")
            return "keyword"
        else:
            # Check if GLiNER is available
            try:
                import gliner
                console.print("\n[green]Selected: GLiNER-based extraction[/green]")
                return "gliner"
            except ImportError:
                console.print("\n[red]Error: GLiNER is not installed.[/red]")
                console.print("Please install GLiNER using: pip install gliner transformers torch")
                console.print("\n[yellow]Falling back to keyword-based extraction...[/yellow]")
                return "keyword"

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
def extract_countries_gliner(text: str, gliner_model: GLiNER) -> List[str]:
    """
    Extract country names from text using GLiNER and countryguess.
    All entities detected by GLiNER are processed through countryguess
    to get standardized country names.
    
    Args:
        text: Input text to process
        gliner_model: Initialized GLiNER model
    
    Returns:
        List of unique country names found
    """
    found_countries = set()
    
    try:
        # Use GLiNER to detect entities
        entities = gliner_model.predict_entities(text, ["LOC", "GPE"])
        
        # Process GLiNER results
        for entity in entities:
            if entity["label"] in ["LOC", "GPE"]:
                entity_text = entity["text"].strip()
                
                # Skip empty or very short entities
                if len(entity_text) < 2:
                    continue
                
                # First try direct country match with pycountry
                try:
                    country = pycountry.countries.get(name=entity_text)
                    if country:
                        found_countries.add(country.name)
                        continue
                except (KeyError, AttributeError):
                    pass
                
                # Process through countryguess for all entities
                try:
                    guessed_country = guess_country(entity_text)
                    if guessed_country and isinstance(guessed_country, dict):
                        # Try to get the official country name
                        country_name = guessed_country.get('name_short')
                        if country_name:
                            try:
                                # Verify with pycountry to get official name
                                country = pycountry.countries.get(name=country_name)
                                if country:
                                    found_countries.add(country.name)
                                else:
                                    # If no exact match in pycountry, use the countryguess name
                                    found_countries.add(country_name)
                            except (KeyError, AttributeError):
                                # If pycountry lookup fails, use the countryguess name
                                found_countries.add(country_name)
                except Exception as e:
                    logger.warning(f"Error in countryguess for '{entity_text}': {e}")
                    # If countryguess fails, try one more time with pycountry
                    try:
                        country = pycountry.countries.get(name=entity_text)
                        if country:
                            found_countries.add(country.name)
                    except (KeyError, AttributeError):
                        pass
    
    except Exception as e:
        logger.warning(f"Error in GLiNER processing: {e}")
    
    # Log the found countries for debugging
    if found_countries:
        logger.info(f"Found countries through GLiNER + countryguess: {', '.join(sorted(found_countries))}")
    
    return list(found_countries)

@task
def process_document(document: dict, method: str, gliner_model: GLiNER = None) -> dict:
    """
    Process a single document and extract countries from each page.
    Returns a dictionary with document info and country frequencies.
    
    Args:
        document: Dictionary containing document data
        method: Either "keyword" or "gliner"
        gliner_model: GLiNER model instance (required if method is "gliner")
    """
    doc_name = document["document_name"]
    page_texts = document["page_texts"]
    
    # Store results for each page
    page_results = {}
    all_countries = Counter()
    
    for page_num, text in page_texts.items():
        # Extract countries based on selected method
        if method == "keyword":
            countries = extract_countries_keyword(text)
        else:  # gliner
            if gliner_model is None:
                raise ValueError("GLiNER model must be provided for gliner-based extraction")
            countries = extract_countries_gliner(text, gliner_model)
        
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
        "extraction_method": method
    }

@flow
def process_extracted_data(method: str = None):
    """
    Main flow to process the extracted JSON data and identify countries.
    
    Args:
        method: Either "keyword" or "gliner" to specify the extraction method.
               If None, user will be prompted to select a method.
    """
    console = Console()
    
    # If no method specified, get user input
    if method is None:
        method = get_extraction_method()
    
    logger.info(f"Starting country extraction pipeline using {method} method")
    console.print(f"\n[bold]Starting extraction using {method} method[/bold]")
    
    input_file = Path("output/extracted_data.json")
    output_file = Path(f"output/country_analysis_{method}.json")
    
    if not input_file.exists():
        error_msg = f"Input file not found: {input_file}"
        logger.error(error_msg)
        console.print(f"\n[red]{error_msg}[/red]")
        return
    
    # Initialize GLiNER model if needed
    gliner_model = None
    if method == "gliner":
        try:
            # Import GLiNER only when needed
            from gliner import GLiNER
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            import torch
            
            console.print("\n[yellow]Initializing GLiNER model...[/yellow]")
            logger.info("Initializing GLiNER model...")
            # Using the specified GLiNER model
            model_name = "urchade/gliner_base"  # GLiNER model from Hugging Face
            gliner_model = GLiNER.from_pretrained(model_name)
            logger.info(f"GLiNER model '{model_name}' initialized successfully")
            console.print(f"[green]GLiNER model '{model_name}' initialized successfully[/green]")
        except ImportError as e:
            error_msg = f"Failed to import GLiNER: {e}. Please install required packages using: pip install gliner transformers torch"
            logger.error(error_msg)
            console.print(f"\n[red]{error_msg}[/red]")
            console.print("\n[yellow]Falling back to keyword-based extraction...[/yellow]")
            method = "keyword"
            output_file = Path("output/country_analysis_keyword.json")
        except Exception as e:
            error_msg = f"Failed to initialize GLiNER model: {e}"
            logger.error(error_msg)
            console.print(f"\n[red]{error_msg}[/red]")
            console.print("\n[yellow]Falling back to keyword-based extraction...[/yellow]")
            method = "keyword"
            output_file = Path("output/country_analysis_keyword.json")
    
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
                result = process_document(doc, method, gliner_model)
                results.append(result)
                logger.info(f"Found {result['total_countries_found']} unique countries in {doc['document_name']}")
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        success_msg = f"Analysis complete. Results saved to {output_file}"
        logger.info(success_msg)
        console.print(f"\n[green]{success_msg}[/green]")
        
    except Exception as e:
        error_msg = f"Error processing data: {e}"
        logger.error(error_msg, exc_info=True)
        console.print(f"\n[red]{error_msg}[/red]")

if __name__ == "__main__":
    start_time = time.time()
    logger.info("Starting country extraction process")
    # You can change the algorithm here: "keyword" or "gliner"
    process_extracted_data(method="keyword")
    end_time = time.time()
    logger.info(f"Total execution time: {end_time - start_time:.2f} seconds") 