"""
qdrant_sync.py
==============
Automated real-time background sync from CRM Resume Parser into central Qdrant Vector Database.
- Generates 384-dimensional dense vectors using FastEmbed (BAAI/bge-small-en-v1.5, ONNX Runtime).
- Incremental upsert to shared Qdrant collection 'resumes' on port 6333.
- Non-blocking execution so CRM frontend experiences 0ms latency impact.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding

logger = logging.getLogger("qdrant_sync")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "resumes"
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Shared storage paths with JD Matcher
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
        _embed_model = TextEmbedding(model_name=EMBED_MODEL_NAME)
    return _embed_model


def build_candidate_search_text(data: dict) -> str:
    """
    Synthesizes a high-signal search document for Qdrant matching.
    """
    parts = []

    name = data.get("name")
    if name and name.lower() not in ["unknown", "curriculum vitae", "resume", "profile"]:
        parts.append(f"Candidate: {name}")

    desigs = data.get("designation")
    if desigs and isinstance(desigs, list):
        clean_desigs = [d.replace("Designation |", "").strip() for d in desigs if d]
        if clean_desigs:
            parts.append(f"Designations: {', '.join(clean_desigs[:5])}")

    skills = data.get("skills")
    if skills and isinstance(skills, list):
        parts.append(f"Skills: {', '.join(skills)}")

    tot_exp = data.get("total_experience")
    if tot_exp is not None and tot_exp > 0:
        parts.append(f"Total Experience: {tot_exp} years")

    summary = data.get("summary")
    if summary and isinstance(summary, str):
        clean_summary = " ".join(summary.split())[:500]
        parts.append(f"Summary: {clean_summary}")

    degrees = data.get("degree")
    if degrees and isinstance(degrees, list):
        parts.append(f"Education: {', '.join(degrees)}")

    projects = data.get("projects")
    if projects and isinstance(projects, list):
        clean_projects = " ".join([str(p).strip() for p in projects[:3]])
        if clean_projects:
            parts.append(f"Projects: {clean_projects[:400]}")

    certs = data.get("certification")
    if certs and isinstance(certs, list):
        parts.append(f"Certifications: {', '.join(certs[:4])}")

    return "\n".join(parts)


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


def sync_resume_to_qdrant(
    parsed_record: dict,
    filename: str,
    file_bytes: Optional[bytes] = None
) -> Optional[int]:
    """
    Synchronously or background-indexes a parsed resume into the shared Qdrant database.
    - Copies original resume to jd-matcher/resumes/
    - Writes flat JSON to jd-matcher/flat_json/
    - Embeds text with FastEmbed and upserts point into Qdrant
    """
    try:
        client = get_qdrant_client()
        embed = get_embed_model()

        point_id = get_next_point_id(client)
        parsed_record["id"] = point_id
        parsed_record["file_name"] = filename

        # 1. Copy file to JD Matcher resumes directory if bytes provided
        if file_bytes and JD_MATCHER_RESUMES.exists():
            dest_file = JD_MATCHER_RESUMES / filename
            try:
                with open(dest_file, "wb") as fp:
                    fp.write(file_bytes)
            except Exception as e:
                logger.warning(f"Failed to copy original file to JD Matcher: {e}")

        # 2. Write flat JSON to JD Matcher flat_json directory
        if JD_MATCHER_FLAT_JSON.exists():
            flat_json_path = JD_MATCHER_FLAT_JSON / f"{point_id}_{Path(filename).stem}_flat.json"
            try:
                with open(flat_json_path, "w", encoding="utf-8") as fp:
                    json.dump(parsed_record, fp, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write flat JSON to JD Matcher: {e}")

        # 3. Generate Vector Embedding
        search_text = build_candidate_search_text(parsed_record)
        vector = list(embed.embed([search_text]))[0].tolist()

        # 4. Upsert into Qdrant
        point = models.PointStruct(
            id=point_id,
            vector=vector,
            payload=parsed_record
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
