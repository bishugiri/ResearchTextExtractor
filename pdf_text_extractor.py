import os
import fitz  # PyMuPDF
import pandas as pd
from langdetect import detect
import logging
import json
from prefect import task, flow
import time
from datetime import datetime

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
                            datefmt='%Y-%m-%d %H:%M')  # Changed date format
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

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
        metadata = {
            "total_characters": len(text),
            "total_words": word_count,
            "detected_language": detected_language
        }
        return {
            "document_name": os.path.basename(pdf_path),
            "text": text,
            "metadata": metadata
        }
    return None

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
            logger.info("-" * 80)
            
        pdf_path = os.path.join(input_folder, pdf_file)
        logger.info(f"Processing PDF {i}/{len(pdf_files)}: {pdf_file}")
        record = process_pdf(pdf_path)
        if record:
            records.append(record)
            logger.info(f"Successfully processed {pdf_file}")
    
    if records:
        logger.info("-" * 80)  # Add separator before saving results
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
    else:
        logger.warning("No PDF files processed or no text extracted.")

def extract_text_from_pdfs_in_directory(directory_path: str) -> list[tuple[str, str]]:
    """
    Extract text from all PDF files in a directory.
    
    Args:
        directory_path (str): Path to the directory containing PDF files
        
    Returns:
        list[tuple[str, str]]: List of tuples containing (filename, extracted_text)
    """
    results = []
    logger = logging.getLogger('pdf_extractor')
    
    try:
        # Check if directory exists
        if not os.path.exists(directory_path):
            logger.error(f"Directory {directory_path} does not exist")
            return results
            
        # Get all PDF files in the directory
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        
        # Extract text from each PDF
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory_path, pdf_file)
            text, _ = extract_text_from_pdf(pdf_path)
            if text:
                results.append((pdf_file, text))
                
        return results
        
    except Exception as e:
        logger.error(f"Error processing directory: {e}")
        return results

if __name__ == "__main__":
    try:
        logger.info("Starting PDF extraction process")
        start_time = time.time()
        process_pdfs_in_folder()
        end_time = time.time()
        logger.info("-" * 80)  # Add separator before final timing
        logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True) 