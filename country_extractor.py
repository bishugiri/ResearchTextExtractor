import pycountry
import pandas as pd
from typing import List, Dict, Tuple
import re

def find_countries_in_text(text: str) -> List[Dict[str, str]]:
    """
    Find countries and their ISO codes in the text.
    
    Args:
        text (str): Text to search for countries
        
    Returns:
        List[Dict[str, str]]: List of dictionaries containing country names and ISO codes
    """
    # Get all country names and their variations
    countries = []
    for country in pycountry.countries:
        # Get official name
        countries.append({
            'name': country.name,
            'code': country.alpha_3
        })
        # Get common name if different
        if hasattr(country, 'common_name'):
            countries.append({
                'name': country.common_name,
                'code': country.alpha_3
            })
    
    # Find countries in text
    found_countries = []
    for country in countries:
        # Create a case-insensitive pattern
        pattern = r'\b' + re.escape(country['name']) + r'\b'
        if re.search(pattern, text, re.IGNORECASE):
            found_countries.append(country)
    
    return found_countries

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
            countries = find_countries_in_text(text)
            
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
            print(f"Error processing {file_path}: {str(e)}")
    
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
    print(f"Country table saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    from pdf_text_extractor import extract_text_from_pdfs_in_directory
    
    # First extract text from PDFs
    results = extract_text_from_pdfs_in_directory(".")
    
    # Create document IDs (using index as ID for this example)
    text_files = [(i+1, path) for i, (_, path) in enumerate(results)]
    
    # Create and save country table
    df = create_country_table(text_files)
    save_country_table(df)
    
    # Display the table
    print("\nCountry Table:")
    print(df) 