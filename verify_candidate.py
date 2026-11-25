"""
verify_candidate.py
===================
Quick CLI tool to inspect, verify, and confirm exactly what candidate data 
is stored in flat_json/ and in the Qdrant Vector Database.

Usage:
    python verify_candidate.py "Rohan Mehta"
    python verify_candidate.py "rohan.mehta@gmail.com"
    python verify_candidate.py "9876543210"
    python verify_candidate.py 172
"""

import sys
import json
import argparse
from pathlib import Path
from qdrant_client import QdrantClient

BASE_DIR = Path(__file__).resolve().parent
FLAT_JSON_DIR = BASE_DIR / "flat_json"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "resumes"


def search_flat_json(query: str):
    """Searches flat_json directory for matching candidates."""
    q_low = query.strip().lower()
    matches = []

    for f in sorted(FLAT_JSON_DIR.glob("*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            
            c_id = str(data.get("id", ""))
            name = str(data.get("name", "")).lower()
            email = str(data.get("email", "")).lower()
            phone = str(data.get("mobile_number", ""))
            fname = f.name.lower()

            if (q_low == c_id or 
                q_low in name or 
                q_low in email or 
                q_low in phone.replace(" ", "").replace("-", "") or 
                q_low in fname):
                matches.append((f, data))
        except Exception:
            continue

    return matches


def search_qdrant(query: str):
    """Searches Qdrant for matching candidate points."""
    try:
        client = QdrantClient(url=QDRANT_URL)
        if query.isdigit():
            # Check by point ID
            pts = client.retrieve(COLLECTION_NAME, [int(query)], with_payload=True)
            if pts:
                return pts

        # Scroll and search payload
        q_low = query.strip().lower()
        points, _ = client.scroll(COLLECTION_NAME, limit=2000, with_payload=True)
        results = []
        for pt in points:
            p = pt.payload or {}
            c_id = str(pt.id)
            name = str(p.get("name", "")).lower()
            email = str(p.get("email", "")).lower()
            phone = str(p.get("mobile_number", ""))
            
            if (q_low == c_id or 
                q_low in name or 
                q_low in email or 
                q_low in phone.replace(" ", "").replace("-", "")):
                results.append(pt)
        return results
    except Exception as e:
        return []


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_candidate.py <name | email | phone | id>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"\n{'═' * 65}")
    print(f"🔍 CANDIDATE DATA VERIFICATION: '{query}'")
    print(f"{'═' * 65}")

    # 1. Search flat_json
    flat_matches = search_flat_json(query)
    print(f"\n📂 Flat JSON Records Found: {len(flat_matches)}")
    for f_path, data in flat_matches:
        print(f"\n   📄 File Path : {f_path.name}")
        print(f"   👤 Name      : {data.get('name')}")
        print(f"   🆔 Point ID  : {data.get('id')}")
        print(f"   📧 Email(s)  : {data.get('email')}")
        print(f"   📱 Phone(s)  : {data.get('mobile_number')}")
        print(f"   ⏱️ Experience: {data.get('total_experience')} years")
        print(f"   🛠️ Skills ({len(data.get('skills', []))}): {', '.join(data.get('skills', [])[:10])}{'...' if len(data.get('skills', []))>10 else ''}")
        if data.get("company_details"):
            print(f"   🏢 Companies : {', '.join([c[:40] for c in data.get('company_details', [])[:2]])}")
        if data.get("last_updated_at"):
            print(f"   🕒 Updated At: {data.get('last_updated_at')}")
        if data.get("update_reasons"):
            print(f"   🔄 Diff Audit: {'; '.join(data.get('update_reasons'))}")

    # 2. Search Qdrant
    qdrant_matches = search_qdrant(query)
    print(f"\n⚡ Qdrant Vector DB Points Found: {len(qdrant_matches)}")
    for pt in qdrant_matches:
        p = pt.payload or {}
        print(f"   Point ID #{pt.id} | Name: {p.get('name')} | Email: {p.get('email')} | Phone: {p.get('mobile_number')} | Skills: {len(p.get('skills', []))} | Exp: {p.get('total_experience')} yrs")

    print(f"\n{'═' * 65}\n")


if __name__ == "__main__":
    main()
