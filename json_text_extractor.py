import os
import json
import pandas as pd
from langdetect import detect
import logging
from prefect import task, flow
import time
from datetime import datetime
from pathlib import Path
import shutil

# Create a logger
logger = logging.getLogger('json_extractor')
logger.setLevel(logging.INFO)

# Create logs directory if it doesn't exist
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create a file handler
log_file = os.path.join(log_dir, f"json_extraction_{datetime.now().strftime('%Y%m%d_%H%M')}.log")
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

@task
def extract_text_from_json(json_path):
    logger.info(f"Starting extraction for {json_path}")
    try:
        # Load the outer JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            outer = json.load(f)
            
        if "ocr" not in outer:
            logger.warning(f"Missing top-level key 'ocr' in {json_path}")
            return None, None
            
        # Parse the inner OCR JSON string
        try:
            ocr_payload = json.loads(outer["ocr"])
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON in 'ocr' field of {json_path}")
            return None, None
            
        # Extract page texts from the pages array
        page_texts = {}
        for page in ocr_payload.get("pages", []):
            if isinstance(page, dict) and "index" in page and "markdown" in page:
                page_num = str(page["index"])
                page_texts[page_num] = page["markdown"]
                
        if not page_texts:
            logger.warning(f"No valid pages found in {json_path}")
            return None, None
            
        # Detect language from combined text
        combined_text = ' '.join(page_texts.values())
        detected_language = detect(combined_text)
        
        logger.info(f"Extracted {len(page_texts)} pages from {json_path}")
        return page_texts, detected_language
        
    except Exception as e:
        logger.error(f"Error extracting {json_path}: {e}")
        return None, None

@task
def process_json(json_path):
    page_texts, detected_language = extract_text_from_json(json_path)
    if page_texts:
        # Calculate total words across all pages
        total_words = sum(len(text.split()) for text in page_texts.values())
        if total_words < 50:
            logger.warning(f"Skipping {os.path.basename(json_path)}: less than 50 words ({total_words})")
            return None
            
        metadata = {
            "total_pages": len(page_texts),
            "total_words": total_words,
            "detected_language": detected_language
        }
        
        return {
            "document_name": os.path.basename(json_path),
            "page_texts": page_texts,
            "metadata": metadata
        }
    return None

@task
def cleanup_output_directory(output_folder: Path):
    """Remove all files from the output directory."""
    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(exist_ok=True)
    logger.info(f"Cleaned up output directory: {output_folder}")

@flow
def process_jsons_in_folder():
    logger.info("Starting JSON processing workflow")
    input_folder = Path("input")
    output_folder = Path("output")
    
    # Clean up output directory
    cleanup_output_directory(output_folder)
    
    # Get list of JSON files
    json_files = list(input_folder.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    records = []
    for i, json_path in enumerate(json_files, 1):
        if i > 1:  # Add separator before each JSON except the first one
            logger.info("-" * 80)
            
        logger.info(f"Processing JSON {i}/{len(json_files)}: {json_path.name}")
        record = process_json(str(json_path))
        if record:
            records.append(record)
            logger.info(f"Successfully processed {json_path.name}")
    
    if records:
        logger.info("-" * 80)  # Add separator before saving results
        # Save to JSON
        json_path = output_folder / "extracted_data.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(records)} records to {json_path}")
    else:
        logger.warning("No JSON files processed or no text extracted.")

if __name__ == "__main__":
    try:
        logger.info("Starting JSON extraction process")
        start_time = time.time()
        process_jsons_in_folder()
        end_time = time.time()
        logger.info("-" * 80)  # Add separator before final timing
        logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True) 