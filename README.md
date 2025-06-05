# Research Text Extractor

This project provides a prefect based pipeline for (1) extracting text from pdfs or json files (2) extract country mentioned.

Run pdf_to_text.py for extracting texts from the pdfs in the input folder

Run json_to_text.py for extracting texts from the already existing json file (output from the mistral ocr)

For country extraction there are two approaches:

1. **Keyword-based**: Uses pycountry's country list for exact matches
2. **GLiNER + CountryGuess**: Uses GLiNER to identify locations and CountryGuess to resolve them to countries

## Requirements

- Python 3.10 or 3.11 (recommended for GLiNER compatibility)
- Required packages (see requirements.txt):
  - pymupdf
  - pandas
  - langdetect
  - prefect>=2.0
  - pyarrow
  - countryguess>=0.4.5
  - pycountry>=24.6.1
  - dateparser>=1.2.0
  - transformers>=4.38.2
  - torch>=2.0.0
  - rich>=14.0.0
  - gliner>=0.2.19
  - huggingface_hub>=0.21.4
  - tqdm>=4.67.1
  - onnxruntime>=1.21.1
  - sentencepiece>=0.2.0

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bishugiri/ResearchTextExtractor.git
cd ResearchTextExtractor
```

2. Create and activate a virtual environment:
```bash
# For Python 3.10
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Algorithm Details

1. **Keyword-based**:
   - Uses pycountry's comprehensive country list
   - Performs exact matches against country names
   - Fast and reliable for standard country names
   - May miss variations or informal country names

2. **GLiNER + CountryGuess**:
   - Uses GLiNER to identify location entities in text
   - Only includes locations with confidence score 
   - Processes identified locations through CountryGuess
   - Can detect country mentions in various forms
   - More flexible but requires more computational resources
   - Requires Python 3.10 or 3.11
   - Provides high-quality results by filtering out low-confidence predictions

### Output

The pipeline generates:
1. CSV file: `output/country_extraction_results_{algorithm}.csv`
2. JSON file: `output/country_extraction_results_{algorithm}.json`
3. Detailed logs: `logs/country_extraction_{algorithm}_{timestamp}.log`

## Notes

- The GLiNER algorithm requires Python 3.10 or 3.11 for compatibility
- All algorithms process only English documents
- Results include country ISO3 codes and mention frequencies
- The GLiNER + CountryGuess approach may identify more country mentions but requires more processing time
