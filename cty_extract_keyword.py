"""
Country extraction pipeline using Prefect 2.x
- Reads JSON with OCR'd page texts
- Extracts country names via keyword matching (pycountry)
- Writes per-page CSV + JSON summary
"""

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

# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------

logger = logging.getLogger("country_extractor")
logger.setLevel(logging.INFO)

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"country_extraction_{datetime.now().strftime('%Y%m%d_%H%M')}.log"

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# **FIXED**: removed stray space after "%" which broke %-style formatting
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M"
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ------------------------------------------------------------------------------
# Helper (plain) function – **not** a Prefect task
# ------------------------------------------------------------------------------

def extract_countries_keyword(text: str) -> List[str]:
    """Return a list of unique country names appearing in *text*."""
    text_lower = text.lower()
    return [
        c.name
        for c in pycountry.countries
        if re.search(rf"\b{re.escape(c.name.lower())}\b", text_lower)
    ]

# ------------------------------------------------------------------------------
# Tasks
# ------------------------------------------------------------------------------

@task
def process_document(document: Dict) -> Dict:
    """Extract countries page‑by‑page for a single document."""
    doc_name = document["document_name"]
    page_texts: Dict[str, str] = document["page_texts"]

    page_results: Dict[str, Dict] = {}
    overall_counter: Counter = Counter()

    for page_num, text in page_texts.items():
        countries = extract_countries_keyword(text)
        counter = Counter(countries)
        page_results[page_num] = {
            "countries": dict(counter),
            "total_countries": len(countries),
        }
        overall_counter.update(countries)

    return {
        "document_name": doc_name,
        "total_pages": len(page_texts),
        "total_countries_found": len(overall_counter),
        "countries_by_page": page_results,
        "overall_country_frequencies": dict(overall_counter),
        "extraction_method": "keyword",
    }
#------------------------------------------------------------------------------
'''
@task
def save_to_csv(results: List[Dict], output_file: Path) -> None:
    """Write `[document, page, country, frequency]` rows to *output_file*."""
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document_name", "page_number", "country", "frequency"])
        for doc in results:
            doc_name = doc["document_name"]
            for page_num, data in doc["countries_by_page"].items():
                for country, freq in data["countries"].items():
                    writer.writerow([doc_name, page_num, country, freq])
    logger.info("CSV results saved to %s", output_file)
'''
# ---------------------------------------------------------------------------------
@task
def save_to_csv(results: List[Dict], output_file: Path) -> None:
    """Write `[document, page, country, iso3, frequency]` rows to *output_file*."""
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document_name", "page_number", "country", "iso3", "frequency"])
        for doc in results:
            doc_name = doc["document_name"]
            for page_num, data in doc["countries_by_page"].items():
                for country, freq in data["countries"].items():
                    try:
                        iso3 = pycountry.countries.lookup(country).alpha_3
                    except LookupError:
                        iso3 = ""
                    writer.writerow([doc_name, page_num, country, iso3, freq])
    logger.info("CSV results saved to %s", output_file)



# ------------------------------------------------------------------------------
# Flow
# ------------------------------------------------------------------------------

@flow
def process_extracted_data(
    input_file: Path = Path("output/extracted_data.json"),
    output_json: Path = Path("output/country_analysis_keyword.json"),
    output_csv: Path = Path("output/country_analysis_keyword.csv"),
) -> None:
    """Main driver."""
    console = Console()
    console.print("[bold cyan]Keyword‑based country extraction started[/bold cyan]")

    if not input_file.exists():
        msg = f"Input file not found: {input_file}"
        logger.error(msg)
        console.print(f"[red]{msg}[/red]")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        documents = json.load(f)

    total_docs = len(documents)
    logger.info("Processing %s documents", total_docs)
    console.print(f"[bold]Processing {total_docs} documents…[/bold]")

    futures = [process_document.submit(doc) for doc in documents]
    results: List[Dict] = []
    for fut in futures:
        res = fut.result()
        results.append(res)
        logger.info("Found %s unique countries in %s", res["total_countries_found"], res["document_name"])

    output_json.parent.mkdir(exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    save_to_csv.submit(results, output_csv)

    console.print(
        f"[green]\n✓ Analysis complete – results saved to:\n- {output_json}\n- {output_csv}[/green]"
    )

# ------------------------------------------------------------------------------
# CLI entry‑point
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    start = time.time()
    logger.info("Starting country extraction process")
    process_extracted_data()
    logger.info("Total execution time: %.2f seconds", time.time() - start)
