"""FineWeb-Edu 500M Token Dataset Pipeline (Phase 2).

Extracts and tokenizes 500,000,000 fresh tokens from FineWeb-Edu for continued pretraining:
1. Guaranteed zero overlap with existing train_expanded.jsonl (100k docs) and val.jsonl (500 docs).
2. Carves out a dedicated, frozen 10,000-document validation set:
     data/tokenized/val_500m.bin + val_500m.json (~11.3M tokens)
3. Packs fresh training documents until reaching exactly 500,000,000 tokens:
     data/tokenized/train_500m.bin + train_500m.json
4. Encodes as memory-mapped uint16 binary tokens formatted as [BOS=1] ... [EOS=2].
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Set

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from huggingface_hub import hf_hub_download
import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

import config

BOS_ID = 1
EOS_ID = 2
PARQUET_LOCAL = config.DATA_DIR / "raw" / "fineweb_000.parquet"
VAL_500M_BIN = config.DATA_DIR / "tokenized" / "val_500m.bin"
VAL_500M_JSON = config.DATA_DIR / "tokenized" / "val_500m.json"
TRAIN_500M_BIN = config.DATA_DIR / "tokenized" / "train_500m.bin"
TRAIN_500M_JSON = config.DATA_DIR / "tokenized" / "train_500m.json"

TARGET_VAL_DOCS = 10000
TARGET_TRAIN_TOKENS = 500000000  # 500 Million tokens


def get_sha256(text: str) -> str:
    """Compute SHA-256 hash of stripped UTF-8 document text."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def index_existing_data() -> tuple[Set[str], Set[str]]:
    """Index IDs and hashes from previous runs to guarantee 100% fresh data."""
    seen_ids: Set[str] = set()
    seen_hashes: Set[str] = set()

    # 1. Base validation set (500 docs)
    if config.VAL_DOCUMENTS_FILE.exists():
        print(f"[Deduplication] Indexing {config.VAL_DOCUMENTS_FILE.name}...")
        with open(config.VAL_DOCUMENTS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    doc = json.loads(line)
                    if doc.get("id"):
                        seen_ids.add(doc["id"])
                    seen_hashes.add(get_sha256(doc.get("text", "")))

    # 2. Track A expanded train set (100k docs)
    train_expanded = config.PROCESSED_DATA_DIR / "train_expanded.jsonl"
    if train_expanded.exists():
        print(f"[Deduplication] Indexing {train_expanded.name}...")
        with open(train_expanded, "r", encoding="utf-8") as f:
            for count, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    doc = json.loads(line)
                    if doc.get("id"):
                        seen_ids.add(doc["id"])
                    seen_hashes.add(get_sha256(doc.get("text", "")))
        print(f"  Indexed {count:,} existing training documents.")

    print(f"  Total seen signatures indexed: {len(seen_ids):,} IDs, {len(seen_hashes):,} hashes.")
    return seen_ids, seen_hashes


def main():
    print("=" * 70)
    print("FineWeb-Edu 500M Token Dataset Pipeline (Phase 2)")
    print("=" * 70)

    # 1. Check parquet availability
    if not PARQUET_LOCAL.exists():
        print(f"\n[Download] Local parquet not found at {PARQUET_LOCAL}. Downloading shard 000_00000.parquet...")
        PARQUET_LOCAL.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=config.DATASET_NAME,
            filename="sample/10BT/000_00000.parquet",
            repo_type="dataset",
            local_dir=str(PARQUET_LOCAL.parent)
        )
        print(f"  Downloaded to {downloaded}")

    # 2. Load tokenizer
    print(f"\n[Tokenizer] Loading tokenizer from {config.TOKENIZER_FILE}...")
    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))
    print(f"  Vocabulary size: {tokenizer.get_vocab_size():,} (Frozen)")

    # 3. Index existing data for deduplication
    seen_ids, seen_hashes = index_existing_data()

    # 4. Open Parquet File
    print(f"\n[Parquet] Opening {PARQUET_LOCAL}...")
    pf = pq.ParquetFile(str(PARQUET_LOCAL))
    total_parquet_rows = pf.metadata.num_rows
    print(f"  Total rows available in shard: {total_parquet_rows:,} documents")

    config.TOKENIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 5. Extract Validation Set (10,000 fresh docs)
    val_exists = VAL_500M_BIN.exists() and VAL_500M_JSON.exists()
    if val_exists:
        with open(VAL_500M_JSON, "r", encoding="utf-8") as f:
            val_meta = json.load(f)
        print(f"\n[Validation Set] Found existing val_500m.bin ({val_meta.get('total_tokens', 0):,} tokens). Skipping re-creation.")
    else:
        print(f"\n[Validation Set] Extracting {TARGET_VAL_DOCS:,} fresh documents...")
        t_val_start = time.time()
        val_docs_collected = 0
        val_tokens_total = 0
        val_bytes_total = 0
        val_doc_lens = []

        with open(VAL_500M_BIN, "wb") as f_val:
            for batch in pf.iter_batches(batch_size=2000, columns=["id", "text"]):
                ids = batch["id"].to_pylist()
                texts = batch["text"].to_pylist()

                clean_texts = []
                for doc_id, text in zip(ids, texts):
                    if not text or not isinstance(text, str) or len(text.strip()) < 50:
                        continue
                    content_hash = get_sha256(text)
                    if (doc_id and doc_id in seen_ids) or (content_hash in seen_hashes):
                        continue
                    if doc_id:
                        seen_ids.add(doc_id)
                    seen_hashes.add(content_hash)
                    clean_texts.append(text)

                    if val_docs_collected + len(clean_texts) >= TARGET_VAL_DOCS:
                        break

                if clean_texts:
                    encodings = tokenizer.encode_batch(clean_texts)
                    for enc, txt in zip(encodings, clean_texts):
                        tok_seq = [BOS_ID] + enc.ids + [EOS_ID]
                        arr = np.array(tok_seq, dtype=np.uint16)
                        f_val.write(arr.tobytes())
                        seq_len = len(tok_seq)
                        val_doc_lens.append(seq_len)
                        val_tokens_total += seq_len
                        val_bytes_total += len(txt.encode("utf-8"))
                        val_docs_collected += 1

                if val_docs_collected >= TARGET_VAL_DOCS:
                    break

        dt_val = time.time() - t_val_start
        val_lens_arr = np.array(val_doc_lens, dtype=np.int32)
        val_meta = {
            "split": "val_500m",
            "total_documents": val_docs_collected,
            "total_tokens": int(val_tokens_total),
            "total_bytes": int(val_bytes_total),
            "storage_bytes_uint16": int(val_tokens_total * 2),
            "storage_mb_uint16": round((val_tokens_total * 2) / (1024 * 1024), 2),
            "elapsed_seconds": round(dt_val, 2),
            "tokens_per_second": round(val_tokens_total / max(dt_val, 1e-6), 1),
            "doc_length_mean": round(float(val_lens_arr.mean()), 2),
            "doc_length_median": float(np.median(val_lens_arr)),
        }
        with open(VAL_500M_JSON, "w", encoding="utf-8") as f_meta:
            json.dump(val_meta, f_meta, indent=2)

        print(f"  [Val Complete] {val_docs_collected:,} docs | {val_tokens_total:,} tokens | {val_meta['storage_mb_uint16']} MB ({dt_val:.1f}s)")

    # 6. Extract Training Set (500,000,000 fresh tokens)
    print(f"\n[Training Set] Packing fresh documents to reach {TARGET_TRAIN_TOKENS:,} tokens...")
    t_train_start = time.time()
    train_docs_collected = 0
    train_tokens_total = 0
    train_bytes_total = 0
    train_doc_lens = []
    dup_skipped = 0

    with open(TRAIN_500M_BIN, "wb") as f_train:
        for batch in pf.iter_batches(batch_size=2500, columns=["id", "text"]):
            ids = batch["id"].to_pylist()
            texts = batch["text"].to_pylist()

            clean_texts = []
            for doc_id, text in zip(ids, texts):
                if not text or not isinstance(text, str) or len(text.strip()) < 50:
                    continue
                content_hash = get_sha256(text)
                if (doc_id and doc_id in seen_ids) or (content_hash in seen_hashes):
                    dup_skipped += 1
                    continue
                if doc_id:
                    seen_ids.add(doc_id)
                seen_hashes.add(content_hash)
                clean_texts.append(text)

            if clean_texts:
                encodings = tokenizer.encode_batch(clean_texts)
                for enc, txt in zip(encodings, clean_texts):
                    tok_seq = [BOS_ID] + enc.ids + [EOS_ID]
                    arr = np.array(tok_seq, dtype=np.uint16)
                    f_train.write(arr.tobytes())
                    seq_len = len(tok_seq)
                    train_doc_lens.append(seq_len)
                    train_tokens_total += seq_len
                    train_bytes_total += len(txt.encode("utf-8"))
                    train_docs_collected += 1

                    if train_tokens_total >= TARGET_TRAIN_TOKENS:
                        break

            if train_docs_collected % 25000 == 0 and clean_texts:
                dt = time.time() - t_train_start
                tok_s = train_tokens_total / max(dt, 1e-6)
                pct = (train_tokens_total / TARGET_TRAIN_TOKENS) * 100
                print(
                    f"  [{pct:5.1f}%] {train_docs_collected:,} docs | "
                    f"{train_tokens_total:,}/{TARGET_TRAIN_TOKENS:,} tokens "
                    f"({tok_s/1e3:,.1f}k tok/s) | dups: {dup_skipped:,}"
                )

            if train_tokens_total >= TARGET_TRAIN_TOKENS:
                break

    # If shard 000 was not enough to reach 500M tokens, fetch shard 001
    if train_tokens_total < TARGET_TRAIN_TOKENS:
        print(f"\n[Additional Shard] Shard 000 yielded {train_tokens_total:,} tokens. Fetching shard 001_00000.parquet...")
        shard_001 = hf_hub_download(
            repo_id=config.DATASET_NAME,
            filename="sample/10BT/001_00000.parquet",
            repo_type="dataset",
            local_dir=str(PARQUET_LOCAL.parent)
        )
        pf_1 = pq.ParquetFile(shard_001)
        with open(TRAIN_500M_BIN, "ab") as f_train:
            for batch in pf_1.iter_batches(batch_size=2500, columns=["id", "text"]):
                ids = batch["id"].to_pylist()
                texts = batch["text"].to_pylist()
                clean_texts = []
                for doc_id, text in zip(ids, texts):
                    if not text or not isinstance(text, str) or len(text.strip()) < 50:
                        continue
                    content_hash = get_sha256(text)
                    if (doc_id and doc_id in seen_ids) or (content_hash in seen_hashes):
                        dup_skipped += 1
                        continue
                    if doc_id:
                        seen_ids.add(doc_id)
                    seen_hashes.add(content_hash)
                    clean_texts.append(text)

                if clean_texts:
                    encodings = tokenizer.encode_batch(clean_texts)
                    for enc, txt in zip(encodings, clean_texts):
                        tok_seq = [BOS_ID] + enc.ids + [EOS_ID]
                        arr = np.array(tok_seq, dtype=np.uint16)
                        f_train.write(arr.tobytes())
                        seq_len = len(tok_seq)
                        train_doc_lens.append(seq_len)
                        train_tokens_total += seq_len
                        train_bytes_total += len(txt.encode("utf-8"))
                        train_docs_collected += 1
                        if train_tokens_total >= TARGET_TRAIN_TOKENS:
                            break

                if train_tokens_total >= TARGET_TRAIN_TOKENS:
                    break

    dt_train = time.time() - t_train_start
    train_lens_arr = np.array(train_doc_lens, dtype=np.int32)
    train_meta = {
        "split": "train_500m",
        "total_documents": train_docs_collected,
        "total_tokens": int(train_tokens_total),
        "total_bytes": int(train_bytes_total),
        "bytes_per_token": round(train_bytes_total / max(train_tokens_total, 1), 4),
        "tokens_per_second": round(train_tokens_total / max(dt_train, 1e-6), 1),
        "storage_bytes_uint16": int(train_tokens_total * 2),
        "storage_mb_uint16": round((train_tokens_total * 2) / (1024 * 1024), 2),
        "elapsed_seconds": round(dt_train, 2),
        "duplicates_skipped": dup_skipped,
        "doc_length_mean": round(float(train_lens_arr.mean()), 2),
        "doc_length_median": float(np.median(train_lens_arr)),
        "doc_length_p95": float(np.percentile(train_lens_arr, 95)),
    }
    with open(TRAIN_500M_JSON, "w", encoding="utf-8") as f_meta:
        json.dump(train_meta, f_meta, indent=2)

    print("\n" + "=" * 70)
    print("500M Token Pipeline Complete!")
    print("=" * 70)
    print(f"  Training documents    : {train_docs_collected:,}")
    print(f"  Training tokens       : {train_tokens_total:,}")
    print(f"  Storage size (uint16) : {train_meta['storage_mb_uint16']} MB (~{train_meta['storage_mb_uint16']/1024:.2f} GB)")
    print(f"  Avg tokens / doc      : {train_meta['doc_length_mean']}")
    print(f"  Duplicates skipped    : {dup_skipped:,}")
    print(f"  Tokenization speed    : {train_meta['tokens_per_second']/1e3:,.1f}k tokens/s")
    print(f"  Total elapsed time    : {dt_train:.1f}s ({dt_train/60:.1f} minutes)")
    print(f"  Train shard path      : {TRAIN_500M_BIN}")
    print(f"  Val shard path        : {VAL_500M_BIN}")
    print("=" * 70)


if __name__ == "__main__":
    main()
