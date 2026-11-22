"""Data preparation pipeline for FineWeb-Edu.

Streams a controlled sample of documents from Hugging Face, validates them,
preserves metadata, and performs a reproducible document-level train/val split.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path
from tqdm import tqdm

from datasets import load_dataset

# Add parent directory to path to import central config
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def stream_and_save_documents(
    num_docs: int = config.NUM_DOCUMENTS,
    output_path: Path = config.RAW_DOCUMENTS_FILE,
) -> tuple[int, int]:
    """Stream valid documents from FineWeb-Edu and save as JSON Lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming from Hugging Face: {config.DATASET_NAME} ({config.DATASET_CONFIG})...")
    print(f"Target document count: {num_docs:,}")
    print(f"Saving raw documents to: {output_path.resolve()}\n")

    dataset = load_dataset(
        config.DATASET_NAME,
        name=config.DATASET_CONFIG,
        split=config.DATASET_SPLIT,
        streaming=True,
    )

    valid_count = 0
    skipped_count = 0
    t0 = time.time()

    with open(output_path, "w", encoding="utf-8") as f_out:
        pbar = tqdm(total=num_docs, desc="Downloading Documents", unit="doc")
        for item in dataset:
            text = item.get("text")

            # Validation check: must be a non-empty string with non-whitespace characters
            if not text or not isinstance(text, str) or not text.strip():
                skipped_count += 1
                continue

            record = {
                "id": item.get("id"),
                "url": item.get("url"),
                "dump": item.get("dump"),
                "language": item.get("language"),
                "token_count": item.get("token_count"),
                "score": item.get("score"),
                "int_score": item.get("int_score"),
                "text": text,
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            valid_count += 1
            pbar.update(1)

            if valid_count >= num_docs:
                break
        pbar.close()

    elapsed = time.time() - t0
    print(f"\nFinished streaming {valid_count:,} valid documents in {elapsed:.1f}s ({skipped_count} empty/invalid skipped).")
    return valid_count, skipped_count


def split_documents(
    raw_path: Path = config.RAW_DOCUMENTS_FILE,
    train_path: Path = config.TRAIN_DOCUMENTS_FILE,
    val_path: Path = config.VAL_DOCUMENTS_FILE,
    train_ratio: float = config.TRAIN_SPLIT_RATIO,
    seed: int = config.RANDOM_SEED,
) -> tuple[int, int]:
    """Split raw documents into train and validation sets at the DOCUMENT level."""
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nPerforming document-level train/validation split (ratio {train_ratio:.0%})...")

    # Read document lines to memory-efficient list
    with open(raw_path, "r", encoding="utf-8") as f:
        doc_lines = f.readlines()

    total_docs = len(doc_lines)
    if total_docs == 0:
        raise ValueError(f"No documents found in {raw_path}")

    # Reproducible shuffle at document level
    rng = random.Random(seed)
    indices = list(range(total_docs))
    rng.shuffle(indices)

    num_train = int(total_docs * train_ratio)
    train_indices = set(indices[:num_train])

    num_train_written = 0
    num_val_written = 0

    with open(train_path, "w", encoding="utf-8") as f_train, open(val_path, "w", encoding="utf-8") as f_val:
        for idx, line in enumerate(doc_lines):
            if idx in train_indices:
                f_train.write(line)
                num_train_written += 1
            else:
                f_val.write(line)
                num_val_written += 1

    print(f"  Train documents:      {num_train_written:,} -> {train_path.resolve()}")
    print(f"  Validation documents: {num_val_written:,} -> {val_path.resolve()}")
    return num_train_written, num_val_written


def verify_saved_documents(
    raw_path: Path = config.RAW_DOCUMENTS_FILE,
    train_path: Path = config.TRAIN_DOCUMENTS_FILE,
    val_path: Path = config.VAL_DOCUMENTS_FILE,
) -> None:
    """Verify that all saved JSONL files can be loaded back cleanly without corruption."""
    print("\nVerifying saved files integrity...")
    for label, p in [("Raw", raw_path), ("Train", train_path), ("Validation", val_path)]:
        if not p.exists():
            raise FileNotFoundError(f"{label} file not found: {p}")

        count = 0
        total_chars = 0
        with open(p, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON decode error in {p} at line {line_no}: {e}")
                assert "text" in data and data["text"].strip(), f"Empty text in {p} line {line_no}"
                count += 1
                total_chars += len(data["text"])

        print(f"  [OK] {label:10}: {count:,} documents verified ({total_chars:,} characters total).")


def main():
    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu sample dataset.")
    parser.add_argument("--num-docs", type=int, default=config.NUM_DOCUMENTS, help="Number of documents to stream.")
    parser.add_argument("--train-ratio", type=float, default=config.TRAIN_SPLIT_RATIO, help="Train split proportion.")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed for splitting.")
    args = parser.parse_args()

    stream_and_save_documents(num_docs=args.num_docs)
    split_documents(train_ratio=args.train_ratio, seed=args.seed)
    verify_saved_documents()
    print("\n[SUCCESS] Dataset preparation complete!")


if __name__ == "__main__":
    main()

