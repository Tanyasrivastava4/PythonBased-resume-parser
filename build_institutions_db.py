"""
build_institutions_db.py — run once to build data/institutions_db.json

Merges two public sources into one lookup DB for the education extractor:
  - AISHE colleges dataset (~43,122 rows) — govt survey of Indian colleges
  - UGC-sourced universities list (~976 rows)

Only full institution names are indexed (for flashtext whole-phrase
matching). An earlier version also built an acronym index (matching bare
tokens like "SVIT" or "EIILM" against institutions' generated initials) —
removed after real-resume testing showed every acronym-tier match came back
flagged needs_review anyway, since a unique hit in the DB never proved it
was the CORRECT institution (see: "SVIT" resolving to a real but unrelated
college in Telangana). A flagged-but-wrong guess can still be misused by
anything downstream that doesn't check the flag; a straightforward
"institution not found" cannot be. Full-name matching plus the
keyword-regex fallback in education.py cover the reliable cases; genuine
acronym-only mentions with no institution keyword nearby now correctly
come back as None instead of a risky guess.
"""

import csv
import json
import urllib.request
from pathlib import Path

# Source datasets — both are public, real government-sourced data, mirrored
# on GitHub as plain CSVs (the govt originals are PDFs / scattered portals).
AISHE_COLLEGES_URL = "https://raw.githubusercontent.com/PriyanKishoreMS/colleges-api/master/data/colleges.csv"
UGC_UNIVERSITIES_URL = "https://raw.githubusercontent.com/saptarshimazumdar/UGC_Indian-University-Dataset/master/UGC%20Universities.csv"


def download_if_missing(url: str, local_path: str):
    path = Path(local_path)
    if path.exists():
        print(f"  Using cached {local_path}")
        return
    print(f"  Downloading {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, local_path)
    print(f"  Saved to {local_path}")


def load_colleges(csv_path: str) -> list:
    out = []
    with open(csv_path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            name = row.get('name', '').strip()
            if name:
                out.append({
                    'name': name,
                    'state': row.get('state', '').strip(),
                    'city': row.get('city', '').strip(),
                    'type': 'college',
                })
    return out


def load_ugc_universities(csv_path: str) -> list:
    """UGC-sourced university list — this is the real institution-of-record
    source, closer to the actual ~1,338 UGC count (this particular scrape is
    a few years old, so it undercounts somewhat — worth refreshing from
    ugc.gov.in directly if exact completeness matters)."""
    out = []
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            name = row.get('Name', '').strip()
            if name:
                out.append({
                    'name': name,
                    'state': '',
                    'city': '',
                    'type': 'university',
                })
    return out


def build(colleges_csv: str, ugc_universities_csv: str, out_path: str):
    institutions = load_colleges(colleges_csv) + load_ugc_universities(ugc_universities_csv)

    # Dedupe exact name+state duplicates
    seen = set()
    deduped = []
    for inst in institutions:
        key = (inst['name'].lower(), inst['state'].lower())
        if key not in seen:
            seen.add(key)
            deduped.append(inst)
    institutions = deduped

    full_names = [inst['name'] for inst in institutions]

    db = {
        'full_names': full_names,  # for flashtext exact/full-name matching
        'count': len(institutions),
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False)

    print(f"Built {out_path}")
    print(f"  Institutions total : {len(institutions)}")


if __name__ == "__main__":
    colleges_csv = "data/raw/colleges.csv"
    ugc_csv = "data/raw/ugc_universities.csv"

    print("Fetching source datasets (cached after first run)...")
    download_if_missing(AISHE_COLLEGES_URL, colleges_csv)
    download_if_missing(UGC_UNIVERSITIES_URL, ugc_csv)

    build(
        colleges_csv=colleges_csv,
        ugc_universities_csv=ugc_csv,
        out_path="data/institutions_db.json",
    )


















#"""
#build_institutions_db.py — run once to build data/institutions_db.json
#
#Merges two public sources into one lookup DB for the education extractor:
#  - AISHE colleges dataset (~43,122 rows) — govt survey of Indian colleges
#  - Hipolabs world-universities list, filtered to India (~475 rows)
#
#For each institution, also generates an acronym from its initials
#(e.g. "Sardar Vallabhbhai Institute of Technology" -> "SVIT") and groups
#institutions by acronym, so the extractor can tell a UNIQUE acronym match
#apart from an AMBIGUOUS one (multiple real institutions share the same
#initials — see SVIT: Gujarat vs whatever college Abhijeet actually meant).
#"""
#
#import csv
#import json
#import re
#import urllib.request
#from pathlib import Path
#
#STOPWORDS = {'of', 'the', 'and', 'for', 'in', 'at', '&', 'de', 'a', 'an', 'to'}
#
## Source datasets — both are public, real government-sourced data, mirrored
## on GitHub as plain CSVs (the govt originals are PDFs / scattered portals).
#AISHE_COLLEGES_URL = "https://raw.githubusercontent.com/PriyanKishoreMS/colleges-api/master/data/colleges.csv"
#UGC_UNIVERSITIES_URL = "https://raw.githubusercontent.com/saptarshimazumdar/UGC_Indian-University-Dataset/master/UGC%20Universities.csv"
#
#
#def download_if_missing(url: str, local_path: str):
#    path = Path(local_path)
#    if path.exists():
#        print(f"  Using cached {local_path}")
#        return
#    print(f"  Downloading {url}")
#    path.parent.mkdir(parents=True, exist_ok=True)
#    urllib.request.urlretrieve(url, local_path)
#    print(f"  Saved to {local_path}")
#
#
#def generate_acronym(name: str) -> str:
#    clean = re.sub(r'[\(\)\[\]\.,\-]', ' ', name)
#    words = [w for w in clean.split() if w.lower() not in STOPWORDS]
#    return ''.join(w[0].upper() for w in words if w and w[0].isalpha())
#
#
#def load_colleges(csv_path: str) -> list:
#    out = []
#    with open(csv_path, encoding='utf-8-sig') as f:
#        for row in csv.DictReader(f):
#            name = row.get('name', '').strip()
#            if name:
#                out.append({
#                    'name': name,
#                    'state': row.get('state', '').strip(),
#                    'city': row.get('city', '').strip(),
#                    'type': 'college',
#                })
#    return out
#
#
#def load_ugc_universities(csv_path: str) -> list:
#    """UGC-sourced university list (not the Hipolabs substitute) — this is the
#    real institution-of-record source, closer to the actual ~1,338 UGC count
#    (this particular scrape is a few years old, so it undercounts somewhat —
#    worth refreshing from ugc.gov.in directly if exact completeness matters)."""
#    out = []
#    with open(csv_path, encoding='utf-8') as f:
#        for row in csv.DictReader(f):
#            name = row.get('Name', '').strip()
#            if name:
#                out.append({
#                    'name': name,
#                    'state': '',
#                    'city': '',
#                    'type': 'university',
#                })
#    return out
#
#
#def build(colleges_csv: str, ugc_universities_csv: str, out_path: str):
#    institutions = load_colleges(colleges_csv) + load_ugc_universities(ugc_universities_csv)
#
#    # Dedupe exact name+state duplicates
#    seen = set()
#    deduped = []
#    for inst in institutions:
#        key = (inst['name'].lower(), inst['state'].lower())
#        if key not in seen:
#            seen.add(key)
#            deduped.append(inst)
#    institutions = deduped
#
#    acronym_index = {}
#    for inst in institutions:
#        acr = generate_acronym(inst['name'])
#        if 2 <= len(acr) <= 6:
#            acronym_index.setdefault(acr, []).append(inst)
#
#    full_names = [inst['name'] for inst in institutions]
#
#    db = {
#        'full_names': full_names,          # for flashtext exact/full-name matching
#        'acronym_index': acronym_index,    # acronym -> list of candidate institutions
#        'count': len(institutions),
#    }
#
#    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#    with open(out_path, 'w', encoding='utf-8') as f:
#        json.dump(db, f, ensure_ascii=False)
#
#    unique_acronyms = sum(1 for v in acronym_index.values() if len(v) == 1)
#    ambiguous_acronyms = sum(1 for v in acronym_index.values() if len(v) > 1)
#    print(f"Built {out_path}")
#    print(f"  Institutions total : {len(institutions)}")
#    print(f"  Unique acronyms     : {unique_acronyms}")
#    print(f"  Ambiguous acronyms  : {ambiguous_acronyms}")
#
#
#if __name__ == "__main__":
#    colleges_csv = "data/raw/colleges.csv"
#    ugc_csv = "data/raw/ugc_universities.csv"
#
#    print("Fetching source datasets (cached after first run)...")
#    download_if_missing(AISHE_COLLEGES_URL, colleges_csv)
#    download_if_missing(UGC_UNIVERSITIES_URL, ugc_csv)
#
#    build(
#        colleges_csv=colleges_csv,
#        ugc_universities_csv=ugc_csv,
#        out_path="data/institutions_db.json",
#    )