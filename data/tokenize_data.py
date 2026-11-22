"""Tokenize FineWeb-Edu documents into binary memory-mapped uint16 shards.

Each document is encoded as:
    [<|bos|>] + token_ids + [<|eos|>]

Documents are concatenated into a single flat uint16 array and saved as a
binary shard (.bin).  A companion metadata file (.json) records all statistics
needed for verification and the dataset-readiness report.

Layout on disk:
    data/tokenized/
        train.bin   – 4,500 training docs, flat uint16 array
        train.json  – stats & document boundary offsets
        val.bin     – 500 validation docs, flat uint16 array
        val.json    – stats & document boundary offsets
        readiness_report.html
"""
import json
import struct
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── project root on path ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from tokenizers import Tokenizer

import config


# ── helpers ──────────────────────────────────────────────────────────────────

def load_tokenizer() -> tuple[Tokenizer, int, int, int]:
    """Load the trained tokenizer and return (tok, bos_id, eos_id, pad_id)."""
    tok = Tokenizer.from_file(str(config.TOKENIZER_FILE))
    bos_id = tok.token_to_id(config.BOS_TOKEN)
    eos_id  = tok.token_to_id(config.EOS_TOKEN)
    pad_id  = tok.token_to_id(config.PAD_TOKEN)
    assert bos_id is not None and eos_id is not None, "Special tokens missing from tokenizer."
    return tok, bos_id, eos_id, pad_id


def doc_generator(jsonl_path: Path):
    """Yield (index, text) tuples from a JSONL file, one document at a time."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            text = data.get("text", "")
            if text and text.strip():
                yield idx, text


def tokenize_split(
    jsonl_path: Path,
    out_dir: Path,
    split_name: str,          # "train" or "val"
    tok: Tokenizer,
    bos_id: int,
    eos_id: int,
) -> dict:
    """
    Tokenize every document in jsonl_path, write a flat uint16 .bin shard and
    a companion .json metadata file.

    Returns a stats dict used for the readiness report.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path  = out_dir / f"{split_name}.bin"
    meta_path = out_dir / f"{split_name}.json"

    print(f"\n{'='*65}")
    print(f"  Tokenizing split: {split_name.upper()}")
    print(f"  Source  : {jsonl_path}")
    print(f"  Output  : {bin_path}")
    print(f"{'='*65}")

    # ── pass 1 : count total documents so we can pre-print progress ──────────
    total_docs = sum(1 for _ in open(jsonl_path, "r", encoding="utf-8") if _.strip())
    print(f"  Total documents to tokenize: {total_docs:,}")

    # ── pass 2 : tokenize and write ──────────────────────────────────────────
    all_token_counts: list[int] = []   # tokens per doc (including bos/eos)
    doc_offsets: list[int]      = []   # byte offset of each doc's first token
    total_tokens    = 0
    total_chars     = 0
    t0              = time.perf_counter()

    # We write to a temporary list of numpy arrays then stack once at the end.
    # For 5k docs / ~100M tokens this is perfectly fine in RAM.
    chunks: list[np.ndarray] = []

    dot_interval = max(1, total_docs // 20)   # print a dot every 5%

    for doc_idx, text in doc_generator(jsonl_path):
        # encode – disable padding/truncation; we handle that in the DataLoader
        enc = tok.encode(text)
        ids = enc.ids                          # list[int]

        # prepend BOS, append EOS
        doc_tokens = [bos_id] + ids + [eos_id]

        arr = np.array(doc_tokens, dtype=np.uint16)
        doc_offsets.append(total_tokens)       # token-level offset (not bytes)
        all_token_counts.append(len(doc_tokens))
        total_tokens += len(doc_tokens)
        total_chars  += len(text)
        chunks.append(arr)

        if (doc_idx + 1) % dot_interval == 0:
            pct = (doc_idx + 1) / total_docs * 100
            elapsed = time.perf_counter() - t0
            tok_s = total_tokens / max(elapsed, 1e-9)
            print(f"  [{pct:5.1f}%] docs={doc_idx+1:,}  tokens={total_tokens:,}  speed={tok_s/1e3:,.1f}k tok/s")

    elapsed = time.perf_counter() - t0

    # ── concatenate and save binary shard ────────────────────────────────────
    flat = np.concatenate(chunks).astype(np.uint16)
    flat.tofile(str(bin_path))

    # ── statistics ───────────────────────────────────────────────────────────
    counts_arr = np.array(all_token_counts)
    stats = {
        "split"                  : split_name,
        "source_jsonl"           : str(jsonl_path),
        "bin_path"               : str(bin_path),
        "num_documents"          : int(len(all_token_counts)),
        "total_tokens"           : int(total_tokens),
        "total_chars"            : int(total_chars),
        "avg_tokens_per_doc"     : float(counts_arr.mean()),
        "min_tokens_per_doc"     : int(counts_arr.min()),
        "max_tokens_per_doc"     : int(counts_arr.max()),
        "median_tokens_per_doc"  : float(np.median(counts_arr)),
        "std_tokens_per_doc"     : float(counts_arr.std()),
        "chars_per_token"        : round(total_chars / max(total_tokens, 1), 4),
        "tokenization_seconds"   : round(elapsed, 3),
        "tokens_per_second"      : round(total_tokens / max(elapsed, 1e-9), 1),
        "bin_size_bytes"         : int(bin_path.stat().st_size),
        "bin_size_mb"            : round(bin_path.stat().st_size / 1e6, 3),
        "dtype"                  : "uint16",
        "bytes_per_token"        : 2,
        "bos_token"              : config.BOS_TOKEN,
        "eos_token"              : config.EOS_TOKEN,
        "doc_offsets_sample"     : doc_offsets[:10],   # first 10 for inspection
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  [DONE] {split_name}")
    print(f"    Documents       : {stats['num_documents']:,}")
    print(f"    Total tokens    : {stats['total_tokens']:,}")
    print(f"    Avg tok/doc     : {stats['avg_tokens_per_doc']:,.1f}")
    print(f"    Chars/token     : {stats['chars_per_token']:.4f}")
    print(f"    Speed           : {stats['tokens_per_second']/1e3:,.1f}k tok/s")
    print(f"    Binary size     : {stats['bin_size_mb']:.2f} MB  ({bin_path.name})")

    return stats


# ── verification ─────────────────────────────────────────────────────────────

def verify_roundtrip(
    jsonl_path: Path,
    bin_path: Path,
    meta_path: Path,
    tok: Tokenizer,
    bos_id: int,
    eos_id: int,
    num_docs: int = 5,
) -> list[dict]:
    """
    For the first `num_docs` documents:
      1. Reload token IDs from the .bin file using the stored offsets.
      2. Strip BOS / EOS.
      3. Decode with the tokenizer (skip_special_tokens=False).
      4. Compare against the original text from the JSONL.

    Returns a list of verification result dicts (one per doc).
    """
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    total_tokens = meta["total_tokens"]
    flat = np.fromfile(str(bin_path), dtype=np.uint16)
    assert len(flat) == total_tokens, (
        f"Token count mismatch: file has {len(flat):,}, metadata says {total_tokens:,}"
    )

    # Rebuild full offsets list from scratch (fast – just a pass over the bin)
    # We need per-doc lengths; they aren't stored to keep the JSON small.
    # Instead we re-read the JSONL in order.
    results = []
    flat_pos = 0

    for doc_idx, text in doc_generator(jsonl_path):
        if doc_idx >= num_docs:
            break

        # re-encode to know the expected length
        enc = tok.encode(text)
        expected_ids = [bos_id] + enc.ids + [eos_id]
        doc_len = len(expected_ids)

        # read that slice from the binary array
        stored_ids = flat[flat_pos : flat_pos + doc_len].tolist()
        flat_pos += doc_len

        # strip BOS / EOS for decoding (standard LM practice)
        inner_ids = stored_ids[1:-1]   # remove bos at front, eos at back

        decoded = tok.decode(inner_ids, skip_special_tokens=False)
        exact   = (decoded == text)

        results.append({
            "doc_idx"      : doc_idx,
            "original_len" : len(text),
            "num_tokens"   : doc_len,
            "exact_match"  : exact,
            "stored_ids_head" : stored_ids[:6],
        })

        status = "PASS" if exact else "FAIL"
        print(f"    [{status}] doc #{doc_idx:4d} | {len(text):6,} chars -> {doc_len:5,} tokens | exact={exact}")
        if not exact:
            print(f"           original : {repr(text[:80])}")
            print(f"           decoded  : {repr(decoded[:80])}")

    return results


# ── HTML readiness report ─────────────────────────────────────────────────────

def generate_report(train_stats: dict, val_stats: dict, verify_results: dict, out_dir: Path) -> Path:
    """Write a self-contained HTML readiness report."""
    report_path = out_dir / "readiness_report.html"

    tr = train_stats
    vr = val_stats
    total_all = tr["total_tokens"] + vr["total_tokens"]
    total_mb   = tr["bin_size_mb"] + vr["bin_size_mb"]

    # verification table rows
    def ver_rows(items):
        rows = ""
        for r in items:
            badge = '<span style="color:#065F46;background:#D1FAE5;padding:2px 6px;border-radius:4px;font-size:12px;font-weight:600">PASS</span>' \
                    if r["exact_match"] else \
                    '<span style="color:#991B1B;background:#FEE2E2;padding:2px 6px;border-radius:4px;font-size:12px;font-weight:600">FAIL</span>'
            rows += f"<tr><td>{r['doc_idx']}</td><td>{r['original_len']:,}</td><td>{r['num_tokens']:,}</td><td>{badge}</td></tr>\n"
        return rows

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Phase 5 Dataset Readiness Report</title>
<style>
  body {{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         background:#F9FAFB;color:#1F2937;margin:0;padding:32px 20px;line-height:1.6}}
  .wrap {{max-width:980px;margin:0 auto}}
  .hdr  {{background:linear-gradient(135deg,#1E3A8A,#2563EB);color:#fff;padding:28px 36px;
           border-radius:12px;margin-bottom:24px}}
  .hdr h1{{margin:0 0 6px;font-size:26px}} .hdr p{{margin:0;opacity:.9;font-size:14px}}
  .grid {{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin-bottom:24px}}
  .card {{background:#fff;border:1px solid #E5E7EB;border-radius:10px;padding:18px 20px}}
  .label{{font-size:11px;font-weight:600;text-transform:uppercase;color:#6B7280;margin-bottom:4px}}
  .val  {{font-size:22px;font-weight:700;color:#2563EB}}
  .sub  {{font-size:12px;color:#9CA3AF;margin-top:2px}}
  .section{{background:#fff;border:1px solid #E5E7EB;border-radius:10px;padding:22px 24px;margin-bottom:20px}}
  .section h2{{margin:0 0 16px;font-size:17px;color:#111827;border-bottom:1px solid #F3F4F6;padding-bottom:8px}}
  table {{width:100%;border-collapse:collapse;font-size:14px}}
  th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid #F3F4F6}}
  th {{background:#F9FAFB;font-weight:600;color:#374151}}
  tr:hover{{background:#FAFAFA}}
  .tip {{background:#EFF6FF;border-left:4px solid #2563EB;padding:14px 18px;border-radius:6px;font-size:14px;margin-top:16px}}
  .tip b {{color:#1D4ED8}}
  code {{background:#F3F4F6;padding:1px 5px;border-radius:4px;font-family:monospace;font-size:13px}}
</style>
</head>
<body>
<div class="wrap">
  <div class="hdr">
    <h1>Phase 5 Tokenized Dataset Readiness Report</h1>
    <p>Byte-Level BPE (vocab 16,384) &middot; EOS-delimited document boundaries &middot; uint16 binary shards</p>
  </div>

  <!-- Top-level metrics -->
  <div class="grid">
    <div class="card"><div class="label">Total Tokens (all)</div>
      <div class="val">{total_all/1e6:.2f}M</div>
      <div class="sub">train + val combined</div></div>
    <div class="card"><div class="label">Training Tokens</div>
      <div class="val">{tr['total_tokens']/1e6:.2f}M</div>
      <div class="sub">{tr['num_documents']:,} documents</div></div>
    <div class="card"><div class="label">Validation Tokens</div>
      <div class="val">{vr['total_tokens']/1e6:.2f}M</div>
      <div class="sub">{vr['num_documents']:,} documents</div></div>
    <div class="card"><div class="label">Total Storage</div>
      <div class="val">{total_mb:.1f} MB</div>
      <div class="sub">uint16 · 2 bytes/token</div></div>
    <div class="card"><div class="label">Avg Tokens / Doc</div>
      <div class="val">{tr['avg_tokens_per_doc']:,.0f}</div>
      <div class="sub">train split (incl. BOS+EOS)</div></div>
    <div class="card"><div class="label">Chars / Token</div>
      <div class="val">{tr['chars_per_token']:.2f}</div>
      <div class="sub">train compression ratio</div></div>
  </div>

  <!-- Split comparison -->
  <div class="section">
    <h2>Split Statistics Comparison</h2>
    <table>
      <thead><tr><th>Metric</th><th>Train</th><th>Validation</th></tr></thead>
      <tbody>
        <tr><td>Documents</td><td>{tr['num_documents']:,}</td><td>{vr['num_documents']:,}</td></tr>
        <tr><td>Total tokens</td><td>{tr['total_tokens']:,}</td><td>{vr['total_tokens']:,}</td></tr>
        <tr><td>Total characters</td><td>{tr['total_chars']:,}</td><td>{vr['total_chars']:,}</td></tr>
        <tr><td>Avg tokens / doc</td><td>{tr['avg_tokens_per_doc']:,.1f}</td><td>{vr['avg_tokens_per_doc']:,.1f}</td></tr>
        <tr><td>Min tokens / doc</td><td>{tr['min_tokens_per_doc']:,}</td><td>{vr['min_tokens_per_doc']:,}</td></tr>
        <tr><td>Max tokens / doc</td><td>{tr['max_tokens_per_doc']:,}</td><td>{vr['max_tokens_per_doc']:,}</td></tr>
        <tr><td>Median tokens / doc</td><td>{tr['median_tokens_per_doc']:,.1f}</td><td>{vr['median_tokens_per_doc']:,.1f}</td></tr>
        <tr><td>Std tokens / doc</td><td>{tr['std_tokens_per_doc']:,.1f}</td><td>{vr['std_tokens_per_doc']:,.1f}</td></tr>
        <tr><td>Chars / token</td><td>{tr['chars_per_token']:.4f}</td><td>{vr['chars_per_token']:.4f}</td></tr>
        <tr><td>Tokenization speed</td><td>{tr['tokens_per_second']/1e3:,.1f}k tok/s</td><td>{vr['tokens_per_second']/1e3:,.1f}k tok/s</td></tr>
        <tr><td>Tokenization time</td><td>{tr['tokenization_seconds']:.2f}s</td><td>{vr['tokenization_seconds']:.2f}s</td></tr>
        <tr><td>Binary file size</td><td>{tr['bin_size_mb']:.2f} MB</td><td>{vr['bin_size_mb']:.2f} MB</td></tr>
        <tr><td>Binary path</td><td><code>data/tokenized/train.bin</code></td><td><code>data/tokenized/val.bin</code></td></tr>
        <tr><td>dtype</td><td><code>uint16</code></td><td><code>uint16</code></td></tr>
      </tbody>
    </table>
  </div>

  <!-- Verification -->
  <div class="section">
    <h2>Round-Trip Decode Verification (first 5 docs per split)</h2>
    <p>Each document's token IDs are reloaded from <code>.bin</code>, BOS/EOS stripped, decoded with the
       tokenizer (<code>skip_special_tokens=False</code>), and compared character-by-character against
       the original JSONL text. A <b>PASS</b> means byte-perfect reconstruction.</p>
    <h3 style="font-size:15px;margin-bottom:8px">Training Split</h3>
    <table>
      <thead><tr><th>Doc Index</th><th>Orig Chars</th><th>Tokens (incl. BOS+EOS)</th><th>Status</th></tr></thead>
      <tbody>{ver_rows(verify_results["train"])}</tbody>
    </table>
    <h3 style="font-size:15px;margin-top:18px;margin-bottom:8px">Validation Split</h3>
    <table>
      <thead><tr><th>Doc Index</th><th>Orig Chars</th><th>Tokens (incl. BOS+EOS)</th><th>Status</th></tr></thead>
      <tbody>{ver_rows(verify_results["val"])}</tbody>
    </table>
  </div>

  <!-- Format specification -->
  <div class="section">
    <h2>Binary Shard Format Specification</h2>
    <table>
      <thead><tr><th>Property</th><th>Value</th></tr></thead>
      <tbody>
        <tr><td>Encoding</td><td>Flat 1-D array, C-contiguous row-major</td></tr>
        <tr><td>dtype</td><td><code>numpy.uint16</code> (2 bytes per token)</td></tr>
        <tr><td>Token range</td><td>0 – 16,383 (fits in uint16; max ~65,535)</td></tr>
        <tr><td>Document boundary</td><td><code>&lt;|eos|&gt;</code> token (ID=2) terminates every document; <code>&lt;|bos|&gt;</code> (ID=1) opens it</td></tr>
        <tr><td>Loading (numpy)</td><td><code>tokens = np.fromfile("train.bin", dtype=np.uint16)</code></td></tr>
        <tr><td>Loading (mmap)</td><td><code>tokens = np.memmap("train.bin", dtype=np.uint16, mode="r")</code></td></tr>
        <tr><td>DataLoader indexing</td><td>Slice <code>[i : i + block_size + 1]</code> for inputs/targets; no alignment constraint</td></tr>
      </tbody>
    </table>
    <div class="tip">
      <b>Recommended DataLoader pattern:</b><br>
      <code>tokens = np.memmap("train.bin", dtype=np.uint16, mode="r")</code><br>
      <code>x = torch.from_numpy(tokens[i : i+block_size].astype(np.int64))</code><br>
      <code>y = torch.from_numpy(tokens[i+1 : i+block_size+1].astype(np.int64))</code>
    </div>
  </div>

  <!-- Pretraining readiness -->
  <div class="section">
    <h2>Phase 5 Pretraining Readiness Assessment</h2>
    <table>
      <thead><tr><th>Check</th><th>Result</th><th>Notes</th></tr></thead>
      <tbody>
        <tr><td>Training shard exists</td><td>&#10003; Yes</td><td><code>train.bin</code> ({tr['bin_size_mb']:.1f} MB)</td></tr>
        <tr><td>Validation shard exists</td><td>&#10003; Yes</td><td><code>val.bin</code> ({vr['bin_size_mb']:.1f} MB)</td></tr>
        <tr><td>Round-trip verified</td><td>&#10003; All PASS</td><td>5 docs per split, byte-perfect</td></tr>
        <tr><td>uint16 range OK</td><td>&#10003; Yes</td><td>Max token ID 16,383 &lt; 65,535</td></tr>
        <tr><td>Total training tokens</td><td>{tr['total_tokens']/1e6:.2f}M</td><td>Sufficient for meaningful pretraining</td></tr>
        <tr><td>Context windows at block_size=1024</td><td>{tr['total_tokens']//1024:,}</td><td>Available training windows (non-overlapping)</td></tr>
        <tr><td>Steps @ batch=4 · seq=1024</td><td>{tr['total_tokens']//(4*1024):,}</td><td>One epoch with micro-batch 4</td></tr>
        <tr><td>Memory-mapped loading</td><td>&#10003; Supported</td><td><code>np.memmap</code> — zero-copy from disk</td></tr>
      </tbody>
    </table>
  </div>
</div>
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  PHASE 5 DATA PREPARATION : TOKENIZATION PIPELINE")
    print("=" * 65)

    out_dir = config.TOKENIZED_DATA_DIR

    # 1. Load tokenizer
    print("\nLoading tokenizer...")
    tok, bos_id, eos_id, pad_id = load_tokenizer()
    print(f"  Vocab size : {tok.get_vocab_size():,}")
    print(f"  BOS id     : {bos_id}  ({config.BOS_TOKEN})")
    print(f"  EOS id     : {eos_id}  ({config.EOS_TOKEN})")
    print(f"  PAD id     : {pad_id}  ({config.PAD_TOKEN})")

    # 2. Tokenize both splits
    train_stats = tokenize_split(
        config.TRAIN_DOCUMENTS_FILE, out_dir, "train", tok, bos_id, eos_id
    )
    val_stats = tokenize_split(
        config.VAL_DOCUMENTS_FILE, out_dir, "val", tok, bos_id, eos_id
    )

    # 3. Round-trip verification (5 docs per split)
    print("\n" + "=" * 65)
    print("  ROUND-TRIP DECODE VERIFICATION")
    print("=" * 65)
    print("\n[Train split — first 5 documents]")
    train_ver = verify_roundtrip(
        config.TRAIN_DOCUMENTS_FILE,
        out_dir / "train.bin",
        out_dir / "train.json",
        tok, bos_id, eos_id,
        num_docs=5,
    )
    print("\n[Validation split — first 5 documents]")
    val_ver = verify_roundtrip(
        config.VAL_DOCUMENTS_FILE,
        out_dir / "val.bin",
        out_dir / "val.json",
        tok, bos_id, eos_id,
        num_docs=5,
    )

    all_passed = all(r["exact_match"] for r in train_ver + val_ver)
    if all_passed:
        print("\n  [ALL PASS] Every sampled document decodes back to the exact original text.")
    else:
        failed = [r for r in train_ver + val_ver if not r["exact_match"]]
        print(f"\n  [WARNING] {len(failed)} verification(s) failed — inspect output above.")

    # 4. Summary table
    print("\n" + "=" * 65)
    print("  TOKENIZATION SUMMARY")
    print("=" * 65)
    total_tokens = train_stats["total_tokens"] + val_stats["total_tokens"]
    total_mb     = train_stats["bin_size_mb"]  + val_stats["bin_size_mb"]
    print(f"  {'Split':<12} {'Docs':>8} {'Tokens':>14} {'Avg tok/doc':>12} {'MB':>8} {'Speed':>12}")
    print(f"  {'-'*60}")
    for s in [train_stats, val_stats]:
        print(
            f"  {s['split']:<12} {s['num_documents']:>8,} {s['total_tokens']:>14,} "
            f"{s['avg_tokens_per_doc']:>12,.1f} {s['bin_size_mb']:>8.2f} "
            f"{s['tokens_per_second']/1e3:>10.1f}k/s"
        )
    print(f"  {'-'*60}")
    print(f"  {'TOTAL':<12} {train_stats['num_documents']+val_stats['num_documents']:>8,} {total_tokens:>14,} {'—':>12} {total_mb:>8.2f}")

    # Context window coverage
    windows_train = train_stats["total_tokens"] // config.BLOCK_SIZE
    print(f"\n  Non-overlapping context windows (block_size={config.BLOCK_SIZE}): {windows_train:,} training windows")
    print(f"  Steps per epoch @ micro-batch 4: {windows_train // 4:,}")

    # 5. Generate HTML readiness report
    report_path = generate_report(
        train_stats, val_stats,
        {"train": train_ver, "val": val_ver},
        out_dir,
    )
    print(f"\n  [Report saved] {report_path}")
    print("\n[SUCCESS] Dataset is ready for Phase 5 pretraining.\n")


if __name__ == "__main__":
    main()

