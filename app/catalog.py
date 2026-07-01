import json
from pathlib import Path


KEYS_ABBREV = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}


def load_catalog():
    catalog_path = Path("data/catalog.json")

    if not catalog_path.exists():
        raise FileNotFoundError(
            "catalog.json not found. Please place it at data/catalog.json."
        )

    with open(catalog_path, "r", encoding="utf-8") as file:
        content = file.read()

    raw_data = json.loads(content, strict=False)
    
    catalog = []

    for entry in raw_data:
        if entry.get("status") != "ok":
            continue

        keys = entry.get("keys", [])

        abbreviations = []
        for key in keys:
            if key in KEYS_ABBREV:
                abbreviations.append(KEYS_ABBREV[key])

        test_type = ",".join(abbreviations)

        item = {
            "entity_id": str(entry.get("entity_id", "")),
            "name": entry.get("name", ""),
            "url": entry.get("link", ""),
            "test_type": test_type,
            "keys": keys,
            "job_levels": entry.get("job_levels", []),
            "languages": entry.get("languages", []),
            "duration": entry.get("duration", ""),
            "remote": entry.get("remote", ""),
            "adaptive": entry.get("adaptive", ""),
            "description": entry.get("description", ""),
        }

        catalog.append(item)

    print(f"Catalog loaded: {len(catalog)} assessments available.")
    return catalog


CATALOG = load_catalog()