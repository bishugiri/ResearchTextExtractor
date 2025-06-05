"""
Country extraction pipeline (GLiNER‑only)
------------------------------------------------
• Reads JSON with OCR page texts (output/extracted_data.json)
• Uses GLiNER ("urchade/gliner_base") to find location/GPE entities
• Converts each entity to a country name with `countryguess` + `pycountry`
• Aggregates counts per‑page and per‑document
• Saves JSON summary and CSV table

Run:
    python country_extractor_gliner.py                     # default paths

All heavy lifting happens inside a single Prefect task per document, so we
load the GLiNER model just once per worker process.
"""

from __future__ import annotations

from prefect import task, flow
import logging
import json
import time
from datetime import datetime
from typing import List, Dict
from pathlib import Path
from collections import Counter

from rich.console import Console

import pycountry
from countryguess import guess_country
from gliner import GLiNER

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("country_extractor")
logger.setLevel(logging.INFO)
log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"country_extraction_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M")
for h in (logging.FileHandler(log_file), logging.StreamHandler()):
    h.setFormatter(fmt); logger.addHandler(h)

# -----------------------------------------------------------------------------
# GLiNER model loader (cached)
# -----------------------------------------------------------------------------
_MODEL_NAME = "urchade/gliner_base"
_gliner_model: GLiNER | None = None

def get_gliner() -> GLiNER:
    """Load GLiNER only once per process."""
    global _gliner_model
    if _gliner_model is None:
        logger.info("Loading GLiNER model '%s'…", _MODEL_NAME)
        _gliner_model = GLiNER.from_pretrained(_MODEL_NAME)
        logger.info("GLiNER model loaded")
    return _gliner_model

# -----------------------------------------------------------------------------
# Country extraction helpers
# -----------------------------------------------------------------------------

def extract_countries_gliner(text: str) -> List[str]:
    """Return a list of unique country names found in *text* using GLiNER."""
    model = get_gliner()
    found: set[str] = set()

    try:
        entities = model.predict_entities(text, ["LOC", "GPE"])
    except Exception as e:  # model or CUDA hiccup
        logger.warning("GLiNER failed: %s", e)
        return []

    for ent in entities:
        ent_text = ent["text"].strip()
        if len(ent_text) < 2:
            continue
        # 1️⃣ Direct pycountry match
        country = pycountry.countries.get(name=ent_text)
        if country:
            found.add(country.name)
            continue
        # 2️⃣ Fallback to countryguess
        try:
            guess = guess_country(ent_text)
            if isinstance(guess, dict):
                name = guess.get("name_short") or guess.get("name_long")
                if name:
                    country = pycountry.countries.get(name=name)
                    found.add(country.name if country else name)
        except Exception:
            # silent: non‑country locations (cities, regions, etc.)
            pass

    return list(found)

# -----------------------------------------------------------------------------
# Tasks
# -----------------------------------------------------------------------------

@task
def process_document(document: Dict) -> Dict:
    """Extract countries page‑by‑page for one document."""
    doc_name = document["document_name"]
    page_texts: Dict[str, str] = document["page_texts"]

    page_results: Dict[str, Dict] = {}
    overall = Counter()

    for page_num, text in page_texts.items():
        countries = extract_countries_gliner(text)
        cnt = Counter(countries)
        overall.update(countries)
        page_results[page_num] = {
            "countries": dict(cnt),
            "total_countries": len(countries),
        }

    return {
        "document_name": doc_name,
        "total_pages": len(page_texts),
        "total_countries_found": len(overall),
        "countries_by_page": page_results,
        "overall_country_frequencies": dict(overall),
        "extraction_method": "gliner",
    }
#--------------------------------------------------------------------------------
'''
@task
def save_to_csv(results: List[Dict], output_file: Path):
    output_file.parent.mkdir(exist_ok=True)
    import csv
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document", "page", "country", "frequency"])
        for doc in results:
            for page, data in doc["countries_by_page"].items():
                for country, freq in data["countries"].items():
                    writer.writerow([doc["document_name"], page, country, freq])
    logger.info("CSV saved to %s", output_file)
'''
#---------------------------------------------------------------------------

@task
def save_to_csv(results: List[Dict], output_file: Path):
    output_file.parent.mkdir(exist_ok=True)
    import csv
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document", "page", "country", "iso3", "frequency"])
        for doc in results:
            for page, data in doc["countries_by_page"].items():
                for country, freq in data["countries"].items():
                    try:
                        iso3 = pycountry.countries.get(name=country).alpha_3
                    except:
                        iso3 = ""
                    writer.writerow([doc["document_name"], page, country, iso3, freq])
    logger.info("CSV saved to %s", output_file)

# -----------------------------------------------------------------------------
# Flow
# -----------------------------------------------------------------------------

@flow
def process_extracted_data(
    input_path: Path = Path("output/extracted_data.json"),
    json_out: Path = Path("output/country_analysis_gliner.json"),
    csv_out: Path = Path("output/country_analysis_gliner.csv"),
):
    console = Console()
    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_path}[/red]")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        docs = json.load(f)
    console.print(f"[bold]Processing {len(docs)} documents…[/bold]")

    futures = [process_document.submit(doc) for doc in docs]
    results = [f.result() for f in futures]

    json_out.parent.mkdir(exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("JSON saved to %s", json_out)

    save_to_csv.submit(results, csv_out)
    console.print(
        f"[green]\n✓ Done – results:\n  • {json_out}\n  • {csv_out}[/green]"
    )

# -----------------------------------------------------------------------------
# CLI entry‑point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    start = time.time()
    logger.info("Starting GLiNER country extraction")
    process_extracted_data()
    logger.info("Finished in %.2f s", time.time() - start)
