"""Byte-level BPE Tokenizer training on FineWeb-Edu training documents.

Trains a 16,384-vocabulary ByteLevel BPE tokenizer strictly on train.jsonl
(excluding validation data) using Hugging Face tokenizers.
Performs comprehensive round-trip verification across Unicode, code, whitespace,
and special tokens.
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


def text_generator(jsonl_path: Path):
    """Memory-efficient generator yielding document text one-by-one from JSONL."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            text = data.get("text", "")
            if text and text.strip():
                yield text


def train_byte_level_bpe(
    train_file: Path = config.TRAIN_DOCUMENTS_FILE,
    output_dir: Path = config.TOKENIZER_DIR,
    vocab_size: int = config.TOKENIZER_VOCAB_SIZE,
    special_tokens: list[str] = config.SPECIAL_TOKENS,
) -> tuple[Tokenizer, dict]:
    """Train a Byte-Level BPE tokenizer from an iterator of documents."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_file.exists():
        raise FileNotFoundError(f"Training corpus not found: {train_file.resolve()}")

    print("=" * 70)
    print("TRAINING BYTE-LEVEL BPE TOKENIZER")
    print("=" * 70)
    print(f"Training corpus:  {train_file.resolve()}")
    print(f"Target vocab:     {vocab_size:,}")
    print(f"Special tokens:   {special_tokens}")
    print(f"Output directory: {output_dir.resolve()}\n")

    # 1. Initialize Byte-Level BPE components
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    # 2. Count documents in training corpus
    doc_count = sum(1 for _ in open(train_file, "r", encoding="utf-8"))
    print(f"Streaming {doc_count:,} documents through BpeTrainer...")

    t0 = time.perf_counter()
    tokenizer.train_from_iterator(text_generator(train_file), trainer=trainer)
    train_time = time.perf_counter() - t0

    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"\n[OK] Training completed in {train_time:.2f} seconds!")
    print(f"Actual vocabulary size: {actual_vocab_size:,}")

    # 3. Save tokenizer model artifact
    tokenizer_json_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))
    print(f"Saved tokenizer artifact: {tokenizer_json_path.resolve()}")

    # 4. Save metadata configuration
    special_token_map = {tok: tokenizer.token_to_id(tok) for tok in special_tokens}
    metadata = {
        "tokenizer_type": "byte-level-bpe",
        "vocab_size": actual_vocab_size,
        "target_vocab_size": vocab_size,
        "special_tokens": special_tokens,
        "special_token_map": special_token_map,
        "training_corpus": str(train_file.resolve()),
        "training_documents": doc_count,
        "training_duration_seconds": round(train_time, 2),
        "created_at": datetime.datetime.now().isoformat(),
    }
    config_json_path = output_dir / "tokenizer_config.json"
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved tokenizer configuration: {config_json_path.resolve()}")

    # 5. Run verification suite
    verify_tokenizer(tokenizer, special_tokens)

    return tokenizer, metadata


def verify_tokenizer(tokenizer: Tokenizer, special_tokens: list[str]) -> None:
    """Verify round-trip lossless encoding and special token handling."""
    print("\n" + "=" * 70)
    print("RUNNING TOKENIZER VERIFICATION SUITE")
    print("=" * 70)

    # A. Special token IDs
    print("\n1. Special Tokens Check:")
    for st in special_tokens:
        tok_id = tokenizer.token_to_id(st)
        assert tok_id is not None, f"Special token {st} was not assigned an ID!"
        recovered = tokenizer.id_to_token(tok_id)
        assert recovered == st, f"Recovered token {recovered} != expected {st}"
        print(f"   [PASS] {st:10} -> ID {tok_id:4d} (decoded: {recovered})")

    # B. Test cases across different text domains
    test_cases = [
        ("English Prose", "The quick brown fox jumps over the lazy dog. Educational content on FineWeb-Edu."),
        ("Whitespace & Newlines", "Line 1\nLine 2\r\n\tIndented with tab and   three spaces.\n\n\nParagraph 2."),
        ("Punctuation & Symbols", "Punctuation: !@#$%^&*()_+-=[]{}|;':\",./<>?~` © ® ™ ± § ¢ £ ¥ € «» „“"),
        ("Numbers & Math", "Integers: 1234567890, floats: 3.1415926535, scientific: 6.022e23, negative: -42."),
        ("Python Code", "def train_gpt(model, data, lr=3e-4):\n    for x, y in data:\n        loss = model(x, y)\n        loss.backward()\n    return loss.item()"),
        ("Unicode & Accents", "Résumé, café, naïve, façade, über, señor, jalapeño, coöperation."),
        ("Non-Latin Scripts", "Greek: Ἑλλάς, Japanese: こんにちは世界, Chinese: 深度学习模型, Arabic: الذكاء الاصطناعي, Russian: Привет мир."),
        ("Emojis & Symbols", "Deep Learning 🚀 with PyTorch 🔥! Emojis: 🌍 📚 💡 🤖 🧠."),
        ("Mixed Boundary", "Special prompt: <|bos|>Once upon a time in a galaxy far away.<|eos|>"),
    ]

    print("\n2. Round-trip Lossless Checks:")
    all_passed = True
    for category, sample in test_cases:
        encoded = tokenizer.encode(sample)
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
        is_exact = (decoded == sample)
        status = "[PASS]" if is_exact else "[FAIL]"
        if not is_exact:
            all_passed = False
        print(f"   {status} {category:25} ({len(sample)} chars -> {len(encoded.ids)} tokens, ratio {len(sample)/max(1, len(encoded.ids)):.2f}x)")
        if not is_exact:
            print(f"      Original: {repr(sample)}")
            print(f"      Decoded:  {repr(decoded)}")

    assert all_passed, "One or more tokenizer round-trip tests failed!"
    print("\n[SUCCESS] All verification tests passed flawlessly! Tokenizer is 100% lossless.")


def main():
    parser = argparse.ArgumentParser(description="Train Byte-Level BPE Tokenizer on FineWeb-Edu.")
    parser.add_argument("--train-file", type=str, default=str(config.TRAIN_DOCUMENTS_FILE), help="Path to train.jsonl.")
    parser.add_argument("--output-dir", type=str, default=str(config.TOKENIZER_DIR), help="Output directory for tokenizer.")
    parser.add_argument("--vocab-size", type=int, default=config.TOKENIZER_VOCAB_SIZE, help="Target vocabulary size.")
    args = parser.parse_args()

    train_byte_level_bpe(
        train_file=Path(args.train_file),
        output_dir=Path(args.output_dir),
        vocab_size=args.vocab_size,
    )


if __name__ == "__main__":
    # Ensure UTF-8 console output
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
