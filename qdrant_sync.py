"""
qdrant_sync.py
==============
Automated real-time background sync from CRM Resume Parser into central Qdrant Vector Database.
- Generates 384-dimensional dense vectors using FastEmbed (BAAI/bge-small-en-v1.5, ONNX Runtime).
- Incremental upsert to shared Qdrant collection 'resumes' on port 6333.
- Non-blocking execution so CRM frontend experiences 0ms latency impact.
"""

import os
import re
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding

from candidate_diff import (
    compare_and_merge_records,
    deep_diff_records,
    normalize_email,
    normalize_phone,
    normalize_emails,
    normalize_phones
)

logger = logging.getLogger("qdrant_sync")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "resumes"
EMBED_MODEL_NAME = "jinaai/jina-embeddings-v2-small-en"

# Thread lock to guarantee atomic check-and-upsert during parallel batch uploads
_sync_lock = threading.Lock()

# Storage paths
ATS_PARSER_DIR = Path(__file__).resolve().parent
ATS_PARSER_RESUMES = ATS_PARSER_DIR / "resumes"
ATS_PARSER_FLAT_JSON = ATS_PARSER_DIR / "flat_json"

JD_MATCHER_DIR = Path("/home/intileo/Desktop/jd-matcher")
JD_MATCHER_RESUMES = JD_MATCHER_DIR / "resumes"
JD_MATCHER_FLAT_JSON = JD_MATCHER_DIR / "flat_json"

_qdrant_client: Optional[QdrantClient] = None
_embed_model: Optional[TextEmbedding] = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL)
    return _qdrant_client


def get_embed_model() -> TextEmbedding:
    global _embed_model
    if _embed_model is None:
        _embed_model = TextEmbedding(model_name=EMBED_MODEL_NAME, max_length=2048)
        _embed_model.model.tokenizer.enable_truncation(max_length=2048)
    return _embed_model


def build_candidate_search_text(data: dict) -> str:
    """
    Synthesizes a structured, high-signal search document for Qdrant matching.
    Utilizes jinaai/jina-embeddings-v2-small-en (8,192 token limit) to preserve:
    - Identity & Clean Titles
    - Distinct Companies
    - Total Experience & Degrees
    - Complete Skills Taxonomy
    - Summary (if present)
    - Full Structured Work Experience (Role @ Company, Dates, and all Key Responsibilities/Bullets)
    - Projects & Key Deliverables
    - Certifications
    """
    parts = []

    name = data.get("name")
    if name and name.lower() not in ["unknown", "curriculum vitae", "resume", "profile"]:
        parts.append(f"Candidate: {name}")

    desigs = data.get("designation")
    if desigs and isinstance(desigs, list):
        clean_desigs = []
        for d in desigs:
            if not d:
                continue
            c_d = re.sub(r"^[•●○◦▪▫■□▸►‣➤✓✔➢❖✦★✪\-\*\—\|\s\d\.\)\(]+", "", str(d).replace("Designation |", "")).strip()
            c_d = re.sub(r"\s*\(\s*\)$", "", c_d).strip()
            if c_d and len(c_d) > 2 and c_d.lower() not in ("unknown", "role", "title"):
                if not any(c_d.lower() == ex.lower() for ex in clean_desigs):
                    clean_desigs.append(c_d)
        if clean_desigs:
            parts.append(f"Designations: {', '.join(clean_desigs)}")

    # Extract distinct companies
    companies = []
    we_list = data.get("work_experience") or []
    if isinstance(we_list, list):
        for we in we_list:
            c = (we.get("company") or "").strip()
            if c and len(c) > 2 and c.lower() not in ("unknown company", "none", "null"):
                if not any(c.lower() == ex.lower() for ex in companies):
                    companies.append(c)

    if not companies:
        comp_lines = data.get("company_details") or []
        if isinstance(comp_lines, list):
            for line in comp_lines:
                if "@" in line and not line.startswith("EXPERIENCE"):
                    comp_part = line.split("@", 1)[1].split("(", 1)[0].strip()
                    comp_part = re.sub(r"^[•●○◦▪▫■□▸►‣➤✓✔➢❖✦★✪\-\*\—\|\s]+", "", comp_part).strip()
                    if comp_part and len(comp_part) > 2 and comp_part.lower() not in ("unknown company", "none", "null"):
                        if not any(comp_part.lower() == ex.lower() for ex in companies):
                            companies.append(comp_part)

    if companies:
        parts.append(f"Companies: {', '.join(companies)}")

    tot_exp = data.get("total_experience")
    if tot_exp is not None and tot_exp > 0:
        parts.append(f"Total Experience: {tot_exp} years")

    degrees = data.get("degree")
    if degrees and isinstance(degrees, list):
        parts.append(f"Education: {', '.join(degrees)}")

    skills = data.get("skills")
    if skills and isinstance(skills, list):
        parts.append(f"Skills: {', '.join(skills)}")

    summary = data.get("summary")
    if summary and isinstance(summary, str):
        clean_summary = " ".join(summary.split())
        if clean_summary:
            parts.append(f"Summary: {clean_summary}")

    # Tiered High-Impact Work Experience
    # - Most Recent / Current Job (index 0): Top 3 high-impact bullets
    # - 2nd Recent Job (index 1): Top 2 bullets
    # - Earlier Jobs (index 2+): Top 1 bullet (or title + company)
    exp_entries = []
    if we_list:
        for idx, we in enumerate(we_list):
            r = we.get("role") or ""
            c = we.get("company") or ""
            s = we.get("start") or ""
            e = we.get("end") or ""
            d_str = f" ({s} - {e})" if (s or e) else ""
            header = f"{r} @ {c}{d_str}".strip(" @")
            resp = we.get("responsibilities") or []

            if idx == 0:
                selected_bullets = resp[:3]
            elif idx == 1:
                selected_bullets = resp[:2]
            else:
                selected_bullets = resp[:1]

            if selected_bullets:
                bullets_text = " ".join([b.strip() for b in selected_bullets])
                exp_entries.append(f"• {header}: {bullets_text}")
            elif header:
                exp_entries.append(f"• {header}")
    elif data.get("company_details"):
        for line in data.get("company_details", []):
            if not line.startswith("EXPERIENCE") and not line.startswith("Total experience"):
                exp_entries.append(line)

    if exp_entries:
        parts.append("Work Experience & Key Responsibilities:\n" + "\n".join(exp_entries[:6]))

    # Projects Section (1 bullet per project defined, up to 4 projects)
    projects = data.get("projects")
    if projects and isinstance(projects, list):
        clean_projs = []
        for p in projects:
            p_str = re.sub(r"^[•●○◦▪▫■□▸►‣➤✓✔➢❖✦★✪\-\*\—\|\s]+", "", str(p)).strip()
            if len(p_str) > 8 and not p_str.lower().startswith(("annexure", "project details", "projects", "academic projects")):
                clean_projs.append(p_str)
        if clean_projs:
            parts.append(f"Projects: {'; '.join(clean_projs[:4])}")

    certs = data.get("certification")
    if certs and isinstance(certs, list):
        parts.append(f"Certifications: {', '.join(certs[:5])}")

    return "\n\n".join(parts)


def get_next_point_id(client: QdrantClient) -> int:
    """
    Determines next available integer ID for Qdrant.
    Checks collection point count and existing point IDs.
    """
    try:
        col = client.get_collection(COLLECTION_NAME)
        count = int(col.points_count)
        # Verify if point already exists at count, find first unused id
        candidate_id = max(count + 1, 1)
        while True:
            existing = client.retrieve(COLLECTION_NAME, [candidate_id])
            if not existing:
                return candidate_id
            candidate_id += 1
    except Exception as e:
        logger.warning(f"Error determining next Qdrant ID: {e}")
        return 1


def find_existing_candidate_point(client: QdrantClient, record: dict):
    """Searches Qdrant by ANY of the candidate's Emails or Mobile Numbers."""
    emails = normalize_emails(record.get("email"))
    phones = normalize_phones(record.get("mobile_number"))

    phone_variants = set()
    for ph in phones:
        phone_variants.add(ph)
        if len(ph) == 10:
            phone_variants.add(f"91{ph}")
            phone_variants.add(f"+91{ph}")
        elif len(ph) == 12 and ph.startswith("91"):
            phone_variants.add(ph[2:])

    raw_mobile = str(record.get("mobile_number") or "").strip()
    if raw_mobile:
        phone_variants.add(raw_mobile)

    should_conditions = []
    for em in emails:
        should_conditions.append(models.FieldCondition(key="email", match=models.MatchText(text=em)))
        should_conditions.append(models.FieldCondition(key="email", match=models.MatchValue(value=em)))
        should_conditions.append(models.FieldCondition(key="emails", match=models.MatchValue(value=em)))

    for ph in phone_variants:
        should_conditions.append(models.FieldCondition(key="mobile_number", match=models.MatchText(text=ph)))
        should_conditions.append(models.FieldCondition(key="mobile_number", match=models.MatchValue(value=ph)))
        should_conditions.append(models.FieldCondition(key="phones", match=models.MatchValue(value=ph)))

    if should_conditions:
        try:
            hits, _ = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(should=should_conditions),
                limit=1,
                with_payload=True
            )
            if hits:
                return hits[0]
        except Exception as e:
            logger.debug(f"Identity lookup error: {e}")

    return None


def check_candidate_conflict(
    parsed_record: dict
) -> Tuple[Optional[int], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Checks if a parsed candidate matches an existing record in Qdrant.
    Returns:
        (existing_point_id, existing_payload, diff_dict) if match found
        (None, None, None) if completely new candidate
    """
    client = get_qdrant_client()
    existing_point = find_existing_candidate_point(client, parsed_record)
    if not existing_point:
        return (None, None, None)

    old_record = existing_point.payload or {}
    diff = deep_diff_records(old_record, parsed_record)
    return (existing_point.id, old_record, diff)


def execute_candidate_resolution(
    action: str,
    parsed_record: dict,
    filename: str,
    existing_point_id: Optional[int] = None,
    file_bytes: Optional[bytes] = None
) -> int:
    """
    Executes one of the 3 user-chosen conflict resolution actions:
    1. 'keep_existing': Discards new resume, keeps existing record untouched.
    2. 'keep_both': Generates brand-new Point ID, saves as separate record. Both coexist.
    3. 'override': Overwrites existing Point ID with new resume data & vector.
    """
    with _sync_lock:
        client = get_qdrant_client()
        embed = get_embed_model()

        if action == "keep_existing":
            logger.info(f"✓ [KEEP_EXISTING] Preserved candidate #{existing_point_id}. Discarded new upload {filename}.")
            return existing_point_id or 0

        elif action == "keep_both":
            # Assign a brand new Point ID so both versions exist independently
            point_id = get_next_point_id(client)
            record_to_save = dict(parsed_record)
            record_to_save["id"] = point_id
            record_to_save["file_name"] = filename
            record_to_save["resolution_action"] = "keep_both"
            logger.info(f"✓ [KEEP_BOTH] Created new independent candidate #{point_id} for {filename}.")

        elif action == "override":
            # Overwrite existing candidate profile at existing_point_id
            point_id = existing_point_id or get_next_point_id(client)
            record_to_save = dict(parsed_record)
            record_to_save["id"] = point_id
            record_to_save["file_name"] = filename
            record_to_save["resolution_action"] = "override"
            record_to_save["last_overwritten_at"] = datetime.now(timezone.utc).isoformat()
            logger.info(f"✓ [OVERRIDE] Completely replaced candidate #{point_id} with {filename}.")

        else:
            point_id = existing_point_id or get_next_point_id(client)
            record_to_save = dict(parsed_record)
            record_to_save["id"] = point_id
            record_to_save["file_name"] = filename

        # Ensure emails and phones list variants are tracked for fast lookup
        all_emails = normalize_emails(record_to_save.get("email"))
        all_phones = normalize_phones(record_to_save.get("mobile_number"))
        p_variants = set()
        for p in all_phones:
            p_variants.add(p)
            if len(p) == 10:
                p_variants.add(f"91{p}")
                p_variants.add(f"+91{p}")
            elif len(p) == 12 and p.startswith("91"):
                p_variants.add(p[2:])
        raw_ph = str(record_to_save.get("mobile_number") or "").strip()
        if raw_ph:
            p_variants.add(raw_ph)
        record_to_save["emails"] = list(all_emails)
        record_to_save["phones"] = list(p_variants)

        # 1. Copy original file to resumes directories if bytes provided
        if file_bytes:
            for target_dir in (JD_MATCHER_RESUMES, ATS_PARSER_RESUMES):
                if target_dir.exists():
                    dest_file = target_dir / filename
                    try:
                        with open(dest_file, "wb") as fp:
                            fp.write(file_bytes)
                    except Exception as e:
                        logger.warning(f"Failed to copy original file to {target_dir}: {e}")

        # 2. Write flat JSON to flat_json directories (clean older json for this point_id if overriding)
        for target_json_dir in (JD_MATCHER_FLAT_JSON, ATS_PARSER_FLAT_JSON):
            if target_json_dir.exists():
                flat_json_path = target_json_dir / f"{point_id}_{Path(filename).stem}_flat.json"
                try:
                    for old_f in target_json_dir.glob(f"{point_id}_*_flat.json"):
                        if old_f.name != flat_json_path.name:
                            try:
                                old_f.unlink()
                            except Exception:
                                pass
                    with open(flat_json_path, "w", encoding="utf-8") as fp:
                        json.dump(record_to_save, fp, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"Failed to write flat JSON to {target_json_dir}: {e}")

        # 3. Generate Vector Embedding
        search_text = build_candidate_search_text(record_to_save)
        vector = list(embed.embed([search_text]))[0].tolist()

        # 4. Upsert into Qdrant
        point = models.PointStruct(
            id=point_id,
            vector=vector,
            payload=record_to_save
        )
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )

        logger.info(f"✓ Successfully indexed candidate #{point_id} ({filename}) to Qdrant")
        return point_id


def sync_resume_to_qdrant(
    parsed_record: dict,
    filename: str,
    file_bytes: Optional[bytes] = None
) -> Optional[int]:
    """
    Synchronously or background-indexes a parsed resume into the shared Qdrant database.
    - Uses thread lock (_sync_lock) to prevent race conditions during parallel batch uploads.
    - Smart Identity Check: Matches candidate by Email or Phone
    - If found: performs Smart Diff & Merges skills/experience in-place (same Point ID)
    - If new: allocates next Point ID
    - Copies original resume to jd-matcher/resumes/ and ats_parser/resumes/
    - Writes flat JSON to jd-matcher/flat_json/ and ats_parser/flat_json/
    - Embeds text with FastEmbed and upserts point into Qdrant
    """
    with _sync_lock:
        try:
            client = get_qdrant_client()
            embed = get_embed_model()

            # Step 1: Candidate Identity Resolution & Smart Diff
            existing_point = find_existing_candidate_point(client, parsed_record)

            if existing_point:
                old_record = existing_point.payload or {}
                diff_res = compare_and_merge_records(old_record, parsed_record)
                action = diff_res["action"]
                point_id = existing_point.id

                if action == "IDENTICAL":
                    logger.info(f"✓ [IDENTICAL] Candidate #{point_id} ({filename}) is identical to existing record. Skipped.")
                    return point_id

                elif action == "KEEP_OLD":
                    logger.info(f"✓ [KEEP_OLD] Candidate #{point_id} ({filename}) is less complete than existing record. Kept existing.")
                    return point_id

                elif action == "UPDATE":
                    record_to_save = diff_res["merged_record"]
                    record_to_save["id"] = point_id
                    record_to_save["file_name"] = filename
                    logger.info(f"🔄 [SMART-UPDATE] Updated Candidate #{point_id} ({filename}): {'; '.join(diff_res['reasons'])}")
            else:
                point_id = get_next_point_id(client)
                record_to_save = parsed_record
                record_to_save["id"] = point_id
                record_to_save["file_name"] = filename

            # Ensure emails and phones list variants are tracked for fast lookup
            all_emails = normalize_emails(record_to_save.get("email"))
            all_phones = normalize_phones(record_to_save.get("mobile_number"))
            p_variants = set()
            for p in all_phones:
                p_variants.add(p)
                if len(p) == 10:
                    p_variants.add(f"91{p}")
                    p_variants.add(f"+91{p}")
                elif len(p) == 12 and p.startswith("91"):
                    p_variants.add(p[2:])
            raw_ph = str(record_to_save.get("mobile_number") or "").strip()
            if raw_ph:
                p_variants.add(raw_ph)
            record_to_save["emails"] = list(all_emails)
            record_to_save["phones"] = list(p_variants)

            # 2. Copy original file to resumes directories if bytes provided
            if file_bytes:
                for target_dir in (JD_MATCHER_RESUMES, ATS_PARSER_RESUMES):
                    if target_dir.exists():
                        dest_file = target_dir / filename
                        try:
                            with open(dest_file, "wb") as fp:
                                fp.write(file_bytes)
                        except Exception as e:
                            logger.warning(f"Failed to copy original file to {target_dir}: {e}")

            # 3. Write flat JSON to flat_json directories (and clean older json for this point_id)
            for target_json_dir in (JD_MATCHER_FLAT_JSON, ATS_PARSER_FLAT_JSON):
                if target_json_dir.exists():
                    flat_json_path = target_json_dir / f"{point_id}_{Path(filename).stem}_flat.json"
                    try:
                        for old_f in target_json_dir.glob(f"{point_id}_*_flat.json"):
                            if old_f.name != flat_json_path.name:
                                try:
                                    old_f.unlink()
                                except Exception:
                                    pass
                        with open(flat_json_path, "w", encoding="utf-8") as fp:
                            json.dump(record_to_save, fp, indent=2)
                    except Exception as e:
                        logger.warning(f"Failed to write flat JSON to {target_json_dir}: {e}")

            # 4. Generate Vector Embedding
            search_text = build_candidate_search_text(record_to_save)
            vector = list(embed.embed([search_text]))[0].tolist()

            # 5. Upsert into Qdrant (in-place update if existing point_id)
            point = models.PointStruct(
                id=point_id,
                vector=vector,
                payload=record_to_save
            )
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[point]
            )

            logger.info(f"✓ Successfully synced resume #{point_id} ({filename}) to Qdrant collection '{COLLECTION_NAME}'")
            return point_id

        except Exception as e:
            logger.error(f"❌ Failed to sync resume to Qdrant: {e}", exc_info=True)
            return None
