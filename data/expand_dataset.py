"""FineWeb-Edu Dataset Expansion Pipeline with SHA-256 Deduplication (Top-Up to 100k Docs).

Expands training documents to a target of 100,000 unique documents while:
1. Guaranteeing zero modification to the frozen validation set (val.jsonl, val.bin).
2. Deduplicating against BOTH existing train (including train_expanded.jsonl) and val.jsonl by document ID and SHA-256 text hash.
3. Resuming / appending seamlessly if train_expanded.jsonl already contains documents (e.g. 50k).
4. Tokenizing the full 100k corpus into data/tokenized/train_expanded.bin with uint16 IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
import config


def get_sha256(text: str) -> str:
    """Compute SHA-256 hash of stripped UTF-8 document text."""
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def get_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of an entire binary or text file."""
    if not path.exists():
        return "NON_EXISTENT"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_existing_signatures(
    val_path: Path, train_expanded_path: Path, train_base_path: Path
) -> tuple[Set[str], Set[str], int]:
    """Load IDs and SHA-256 hashes of validation docs and existing training docs."""
    seen_ids: Set[str] = set()
    seen_hashes: Set[str] = set()

    print("[Deduplication] Indexing frozen validation documents...")
    val_count = 0
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc.get("id")
            if doc_id:
                seen_ids.add(doc_id)
            seen_hashes.add(get_sha256(doc.get("text", "")))
            val_count += 1
    print(f"  Indexed {val_count:,} validation documents (IDs & SHA-256 hashes).")

    # Check if train_expanded.jsonl exists
    current_train_count = 0
    if train_expanded_path.exists():
        print(f"[Deduplication] Indexing existing documents in {train_expanded_path.name}...")
        with open(train_expanded_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                doc_id = doc.get("id")
                if doc_id:
                    seen_ids.add(doc_id)
                seen_hashes.add(get_sha256(doc.get("text", "")))
                current_train_count += 1
        print(f"  Indexed {current_train_count:,} documents from {train_expanded_path.name}.")
    else:
        print(f"[Deduplication] Indexing base training documents from {train_base_path.name}...")
        # Write base docs into train_expanded_path first
        train_expanded_path.parent.mkdir(parents=True, exist_ok=True)
        with open(train_base_path, "r", encoding="utf-8") as f_in, open(train_expanded_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                doc_id = doc.get("id")
                if doc_id:
                    seen_ids.add(doc_id)
                seen_hashes.add(get_sha256(doc.get("text", "")))
                f_out.write(line + "\n")
                current_train_count += 1
        print(f"  Initialized {train_expanded_path.name} with {current_train_count:,} base documents.")

    return seen_ids, seen_hashes, current_train_count


def expand_dataset(
    target_train_docs: int = 100000,
    output_jsonl: Path = config.PROCESSED_DATA_DIR / "train_expanded.jsonl",
) -> dict:
    """Stream from FineWeb-Edu and top up unique documents to target_train_docs."""
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    val_hash_before = get_file_sha256(config.VAL_DOCUMENTS_FILE)
    val_bin_hash_before = get_file_sha256(config.VAL_BIN)

    seen_ids, seen_hashes, current_count = load_existing_signatures(
        config.VAL_DOCUMENTS_FILE, output_jsonl, config.TRAIN_DOCUMENTS_FILE
    )

    needed_new = target_train_docs - current_count

    print(f"\n[Expansion Target]")
    print(f"  Current training docs   : {current_count:,}")
    print(f"  Target training docs    : {target_train_docs:,}")
    print(f"  Additional required     : {needed_new:,}")
    print(f"  Output JSONL            : {output_jsonl}")

    if needed_new <= 0:
        print(f"Target {target_train_docs:,} already reached or exceeded (current: {current_count:,}).")
        return {"total_docs": current_count, "new_docs": 0, "duplicates_skipped": 0}

    print(f"\n[Streaming] Connecting to Hugging Face: {config.DATASET_NAME} ({config.DATASET_CONFIG})...")
    stream_ds = load_dataset(
        config.DATASET_NAME,
        name=config.DATASET_CONFIG,
        split=config.DATASET_SPLIT,
        streaming=True,
    )

    new_saved = 0
    dup_skipped = 0
    empty_skipped = 0
    t0 = time.time()

    with open(output_jsonl, "a", encoding="utf-8") as f_out:
        t_stream_start = time.time()
        for item in stream_ds:
            text = item.get("text")

            # Validate text
            if not text or not isinstance(text, str) or not text.strip():
                empty_skipped += 1
                continue

            doc_id = item.get("id")
            content_hash = get_sha256(text)

            # Deduplication against val and all existing train sets
            if (doc_id and doc_id in seen_ids) or (content_hash in seen_hashes):
                dup_skipped += 1
                continue

            # Record seen signatures
            if doc_id:
                seen_ids.add(doc_id)
            seen_hashes.add(content_hash)

            record = {
                "id": doc_id,
                "url": item.get("url"),
                "dump": item.get("dump"),
                "language": item.get("language"),
                "token_count": item.get("token_count"),
                "score": item.get("score"),
                "int_score": item.get("int_score"),
                "text": text,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            new_saved += 1

            if new_saved % 5000 == 0 or (current_count + new_saved) >= target_train_docs:
                elapsed = time.time() - t_stream_start
                rate = new_saved / max(elapsed, 1e-6)
                total_current = current_count + new_saved
                print(
                    f"  [{total_current:,}/{target_train_docs:,} docs] "
                    f"+{new_saved:,} new ({rate:.1f} doc/s) | "
                    f"dups skipped: {dup_skipped:,} | empty: {empty_skipped:,}"
                )

            if (current_count + new_saved) >= target_train_docs:
                break

    total_time = time.time() - t0
    final_docs = current_count + new_saved
    print(f"\n[Expansion Complete]")
    print(f"  Total training docs in {output_jsonl.name}: {final_docs:,}")
    print(f"  New documents added   : {new_saved:,}")
    print(f"  Duplicates skipped    : {dup_skipped:,}")
    print(f"  Empty/invalid skipped : {empty_skipped:,}")
    print(f"  Elapsed time          : {total_time:.1f}s ({total_time / 60:.1f}m)")

    # Integrity verification of frozen validation benchmark
    val_hash_after = get_file_sha256(config.VAL_DOCUMENTS_FILE)
    val_bin_hash_after = get_file_sha256(config.VAL_BIN)
    assert val_hash_before == val_hash_after, "FATAL: val.jsonl was modified!"
    assert val_bin_hash_before == val_bin_hash_after, "FATAL: val.bin was modified!"
    print(f"[Verification] Frozen validation files remain 100% UNCHANGED and byte-identical:")
    print(f"  val.jsonl SHA-256: {val_hash_after[:16]}... (MATCH)")
    print(f"  val.bin   SHA-256: {val_bin_hash_after[:16]}... (MATCH)")

    return {
        "total_docs": final_docs,
        "base_docs": current_count,
        "new_docs": new_saved,
        "duplicates_skipped": dup_skipped,
        "empty_skipped": empty_skipped,
        "elapsed_seconds": round(total_time, 2),
    }


def tokenize_expanded(
    jsonl_path: Path = config.PROCESSED_DATA_DIR / "train_expanded.jsonl",
    output_bin: Path = config.TOKENIZED_DATA_DIR / "train_expanded.bin",
    output_meta: Path = config.TOKENIZED_DATA_DIR / "train_expanded.json",
    batch_size: int = 1000,
) -> dict:
    """Tokenize the expanded JSONL into a memory-mapped uint16 binary shard."""
    output_bin.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  TOKENIZING EXPANDED DATASET (Target: 100,000 Docs)")
    print(f"  Source : {jsonl_path}")
    print(f"  Output : {output_bin}")
    print(f"{'='*65}")

    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))
    bos_id = tokenizer.token_to_id(config.BOS_TOKEN)
    eos_id = tokenizer.token_to_id(config.EOS_TOKEN)

    t0 = time.time()
    total_tokens = 0
    total_bytes = 0
    doc_count = 0
    doc_lens = []

    with open(jsonl_path, "r", encoding="utf-8") as f_in, open(output_bin, "wb") as f_out:
        batch_texts = []

        def flush_batch(texts: list[str]):
            nonlocal total_tokens, total_bytes, doc_count, doc_lens
            encodings = tokenizer.encode_batch(texts)
            for enc, txt in zip(encodings, texts):
                tok_seq = [bos_id] + enc.ids + [eos_id]
                arr = np.array(tok_seq, dtype=np.uint16)
                f_out.write(arr.tobytes())
                seq_len = len(tok_seq)
                doc_lens.append(seq_len)
                total_tokens += seq_len
                total_bytes += len(txt.encode("utf-8"))
                doc_count += 1

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            text = data.get("text", "")
            if text:
                batch_texts.append(text)
                if len(batch_texts) >= batch_size:
                    flush_batch(batch_texts)
                    batch_texts = []
                    if doc_count % 10000 == 0:
                        elapsed = time.time() - t0
                        tok_s = total_tokens / max(elapsed, 1e-6)
                        print(f"  Tokenized {doc_count:,} docs | {total_tokens:,} tokens ({tok_s/1e3:.1f}k tok/s)")

        if batch_texts:
            flush_batch(batch_texts)

    elapsed_tok = time.time() - t0
    doc_lens_arr = np.array(doc_lens, dtype=np.int32)

    meta = {
        "split": "train_expanded",
        "total_documents": doc_count,
        "total_tokens": int(total_tokens),
        "total_bytes": int(total_bytes),
        "bytes_per_token": round(total_bytes / max(total_tokens, 1), 4),
        "tokens_per_second": round(total_tokens / max(elapsed_tok, 1e-6), 1),
        "storage_bytes_uint16": int(total_tokens * 2),
        "storage_mb_uint16": round((total_tokens * 2) / (1024 * 1024), 2),
        "elapsed_seconds": round(elapsed_tok, 2),
        "doc_length_min": int(doc_lens_arr.min()),
        "doc_length_max": int(doc_lens_arr.max()),
        "doc_length_mean": round(float(doc_lens_arr.mean()), 2),
        "doc_length_median": float(np.median(doc_lens_arr)),
        "doc_length_p95": float(np.percentile(doc_lens_arr, 95)),
    }

    with open(output_meta, "w", encoding="utf-8") as f_meta:
        json.dump(meta, f_meta, indent=2)

    print(f"\n[Tokenization Complete]")
    print(f"  Total documents       : {doc_count:,}")
    print(f"  Total tokens          : {total_tokens:,}")
    print(f"  Total UTF-8 bytes     : {total_bytes:,}")
    print(f"  Avg bytes / token     : {meta['bytes_per_token']}")
    print(f"  Avg tokens / doc      : {meta['doc_length_mean']}")
    print(f"  Median tokens / doc   : {meta['doc_length_median']}")
    print(f"  Storage (.bin uint16) : {meta['storage_mb_uint16']} MB")
    print(f"  Throughput            : {meta['tokens_per_second']/1e3:.1f}k tokens/s")
    print(f"  Elapsed time          : {elapsed_tok:.1f}s")
    print(f"  Metadata saved to     : {output_meta}")

    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expand FineWeb-Edu dataset to 100k docs with deduplication.")
    parser.add_argument("--target-docs", type=int, default=100000, help="Target total training documents")
    parser.add_argument("--skip-stream", action="store_true", help="Skip streaming, tokenize existing expanded jsonl")
    args = parser.parse_args()

    jsonl_path = config.PROCESSED_DATA_DIR / "train_expanded.jsonl"
    bin_path = config.TOKENIZED_DATA_DIR / "train_expanded.bin"
    meta_path = config.TOKENIZED_DATA_DIR / "train_expanded.json"

    if not args.skip_stream:
        expand_dataset(target_train_docs=args.target_docs, output_jsonl=jsonl_path)

    tokenize_expanded(jsonl_path=jsonl_path, output_bin=bin_path, output_meta=meta_path)
