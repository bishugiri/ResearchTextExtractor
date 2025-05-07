# requirements: prefect>=2.0, dateparser, pycountry
# pip install prefect dateparser pycountry

from prefect import task, flow
import re
import dateparser
import pycountry
import pandas as pd

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

if __name__ == "__main__":
    # Read the parquet file
    df = pd.read_parquet("extracted_data.parquet")
    results = []
    for idx, row in df.iterrows():
        document_id = row.get("document_id", row.get("document_name", f"doc_{idx}"))
        text = row["text"]
        result = document_iso3_flow(document_id, text)
        results.append(result)
        print(result)  # Print each result

    # Optionally, save all results to a new file
    # pd.DataFrame(results).to_json("iso3_extraction_results.json", orient="records", indent=2)
