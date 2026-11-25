#!/usr/bin/env python3
"""
index_to_qdrant.py
==================
Syncs parsed resume records from ats_parser/flat_json/ into Qdrant Vector Database
and copies original resume files to jd-matcher/resumes/ so they can be viewed/downloaded.

Usage:
    python index_to_qdrant.py
"""

import os
import gc
import sys
import json
import time
import shutil
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding

BASE_DIR = Path(__file__).resolve().parent
FLAT_JSON_DIR = BASE_DIR / "flat_json"
ATS_RESUMES_DIR = BASE_DIR / "resumes"
JD_MATCHER_RESUMES_DIR = Path("/home/intileo/Desktop/jd-matcher/resumes")

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "resumes")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "jinaai/jina-embeddings-v2-small-en")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
BATCH_SIZE = 8
MAX_TOKENS = 2048

from qdrant_sync import build_candidate_search_text


def sync_resumes_to_jd_matcher():
    """Copies any new resume files to jd-matcher/resumes directory."""
    if not JD_MATCHER_RESUMES_DIR.exists():
        JD_MATCHER_RESUMES_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in ATS_RESUMES_DIR.iterdir():
        if src.is_file():
            dst = JD_MATCHER_RESUMES_DIR / src.name
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)
                copied += 1
    if copied > 0:
        print(f"✓ Synced {copied} resume files to jd-matcher/resumes/")


def main():
    print(f"\n{'═' * 65}")
    print(f"🚀 INGESTING PARSED RESUMES INTO QDRANT VECTOR DB")
    print(f"{'═' * 65}")
    print(f"• Source Directory : {FLAT_JSON_DIR}")
    print(f"• Qdrant URL       : {QDRANT_URL}")
    print(f"• Collection Name  : {COLLECTION_NAME}")
    print(f"• Embedding Model  : {EMBED_MODEL_NAME}")
    print(f"{'═' * 65}\n")

    # Sync files to jd-matcher
    sync_resumes_to_jd_matcher()

    # 1. Connect to Qdrant
    print("1. Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL)

    # 2. Re-create collection
    print(f"2. Recreating collection '{COLLECTION_NAME}' (dim=512, Cosine)...")
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=512,
            distance=models.Distance.COSINE
        )
    )

    # 3. Load FastEmbed model
    print(f"3. Loading FastEmbed model on CPU (max_length={MAX_TOKENS})...")
    t_load = time.perf_counter()
    embed_model = TextEmbedding(model_name=EMBED_MODEL_NAME, max_length=MAX_TOKENS)
    embed_model.model.tokenizer.enable_truncation(max_length=MAX_TOKENS)
    print(f"✓ Model loaded in {time.perf_counter() - t_load:.2f}s")

    # 4. Read all flat JSON records
    json_paths = sorted(
        FLAT_JSON_DIR.glob("*.json"),
        key=lambda p: int(p.stem.split("_")[0]) if p.stem.split("_")[0].isdigit() else 999999
    )
    total_files = len(json_paths)
    print(f"\n4. Found {total_files} resume records in {FLAT_JSON_DIR.name}/ to ingest.")

    records_to_process = []
    texts_to_embed = []
    for p in json_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            doc_text = build_candidate_search_text(data)
            records_to_process.append(data)
            texts_to_embed.append(doc_text)
        except Exception as e:
            print(f"Warning: Failed to read {p.name}: {e}")

    # 5. Batch Embed & Upsert
    t_start = time.perf_counter()
    total_upserted = 0
    seen_ids = set()
    max_id = 0
    for r in records_to_process:
        if isinstance(r.get("id"), int):
            max_id = max(max_id, r["id"])

    print(f"\n5. Embedding and uploading in batches of {BATCH_SIZE}...")
    for i in range(0, len(records_to_process), BATCH_SIZE):
        batch_records = records_to_process[i:i + BATCH_SIZE]
        batch_texts = texts_to_embed[i:i + BATCH_SIZE]

        # Generate vectors using FastEmbed ONNX runtime with batch_size=BATCH_SIZE
        embeddings = list(embed_model.embed(batch_texts, batch_size=BATCH_SIZE))

        points = []
        for record, emb in zip(batch_records, embeddings):
            pid = record.get("id")
            if pid is None or not str(pid).isdigit() or int(pid) in seen_ids:
                max_id += 1
                pid = max_id
            else:
                pid = int(pid)
                max_id = max(max_id, pid)
            seen_ids.add(pid)
            record["id"] = pid
            points.append(
                models.PointStruct(
                    id=pid,
                    vector=emb.tolist(),
                    payload=record
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )
        total_upserted += len(points)
        gc.collect()

        if total_upserted % 20 == 0 or total_upserted == len(records_to_process):
            elapsed = time.perf_counter() - t_start
            speed = total_upserted / max(elapsed, 0.01)
            print(f"   ✓ Uploaded {total_upserted}/{len(records_to_process)} candidates ({speed:.1f} docs/s)...")

    total_time = time.perf_counter() - t_start
    print(f"✓ Upserted {total_upserted} points in {total_time:.2f}s ({total_upserted/max(total_time, 0.01):.1f} docs/sec)")

    # 6. Create Payload Indexes
    print("\n6. Creating Payload Indexes for hybrid filtering...")
    index_fields = [
        ("skills", models.PayloadSchemaType.KEYWORD),
        ("total_experience", models.PayloadSchemaType.FLOAT),
        ("degree", models.PayloadSchemaType.KEYWORD),
        ("gender", models.PayloadSchemaType.KEYWORD),
        ("age", models.PayloadSchemaType.FLOAT),
        ("email", models.PayloadSchemaType.KEYWORD),
        ("mobile_number", models.PayloadSchemaType.KEYWORD)
    ]
    for field, schema in index_fields:
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=schema
            )
        except Exception:
            pass
    print("✓ Payload indexes active on 'skills', 'total_experience', 'degree', 'gender', 'age', 'email', 'mobile_number'")

    print(f"\n{'═' * 65}")
    print(f"🎉 QDRANT INGESTION COMPLETE! {total_upserted} RESUMES READY FOR MATCHING")
    print(f"{'═' * 65}\n")


if __name__ == "__main__":
    main()
