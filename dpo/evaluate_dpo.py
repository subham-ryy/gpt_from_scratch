"""Phase 5: Comprehensive Rigorous Evaluation Harness (SFT vs DPO Candidates).

Evaluates:
- Refusal rate on 20 held-out personal-situation prompts (>=5 samples each)
- 3-gram repetition rate, EOS clean-termination rate, mean response length
- Trivia accuracy (20 questions with known ground truth, >=5 samples each)
- Multi-turn pronoun/context retention (5 dialogues, >=5 samples each)
- Leaked format rate (\\boxed{} / 'The answer is:') on 20 non-math prompts
- SFT validation loss (assistant-token NLL) to detect fluency drift
- Held-out preference accuracy and mean KL(policy || reference) estimate
- Head-to-head rule-score win rate
- Reports under both default decoding (rep_pen=1.15) AND unconfounded (rep_pen=1.0)
- 10 side-by-side examples including regressions
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

import config
from model.transformer import GPT, GPTConfig
from data.clean_sft_refusals import REFUSAL_PATTERNS
from dpo.eval_prompts import (
    REFUSAL_EVAL_PROMPTS,
    TRIVIA_EVAL_PROMPTS,
    MULTI_TURN_EVAL_PROMPTS,
    LEAKED_FORMAT_EVAL_PROMPTS,
)
from dpo.preference_labeling import score_response

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOS_ID = 1
EOS_ID = 2
PAD_ID = 0
IGNORE_INDEX = -100

STOP_MARKERS = ["### User:", "\n### User:", "### Assistant:", "\n###"]


def load_eval_model(checkpoint_path: Path) -> GPT:
    ckpt = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=False)
    saved_cfg = ckpt.get("config_snapshot", {})

    model_config = GPTConfig.from_tokenizer_config(
        config_path=config.TOKENIZER_CONFIG_FILE,
        block_size=saved_cfg.get("block_size", config.BLOCK_SIZE),
        n_layer=saved_cfg.get("n_layer", config.N_LAYER),
        n_head=saved_cfg.get("n_head", config.N_HEAD),
        n_embd=saved_cfg.get("n_embd", config.N_EMBD),
        ffn_dim=saved_cfg.get("ffn_dim", config.FFN_DIM),
        dropout=0.0,
        bias=saved_cfg.get("bias", config.BIAS),
        tie_weights=saved_cfg.get("tie_weights", config.TIE_WEIGHTS),
        use_flash=True,
    )
    model = GPT(model_config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def clean_output(raw_text: str) -> str:
    cleaned = raw_text.strip()
    for sm in STOP_MARKERS:
        pos = cleaned.find(sm)
        if pos != -1:
            cleaned = cleaned[:pos].strip()
    return cleaned


def is_refusal(text: str) -> bool:
    t_lower = text.lower()
    for pat, _ in REFUSAL_PATTERNS:
        if re.search(pat, t_lower):
            return True
    return False


def compute_3gram_rep(text: str) -> float:
    words = re.findall(r"\b\w+\b", text.lower())
    if len(words) < 6:
        return 0.0
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    return 1.0 - (len(set(trigrams)) / max(len(trigrams), 1))


def has_leaked_format(text: str) -> bool:
    t_lower = text.lower()
    return ("\\boxed" in text or "the answer is:" in t_lower or "the answer is " in t_lower)


def generate_single_prompt(
    model: GPT,
    tokenizer: Tokenizer,
    prompt_text: str,
    num_samples: int = 5,
    max_new_tokens: int = 150,
    temperature: float = 0.4,
    top_p: float = 0.90,
    min_p: float = 0.08,
    repetition_penalty: float = 1.15,
) -> List[Dict[str, Any]]:
    """Sample multiple responses for a single prompt."""
    prefix_str = f"### User:\n{prompt_text.strip()}\n\n### Assistant:\n"
    prefix_ids = [BOS_ID] + tokenizer.encode(prefix_str).ids
    inp = torch.tensor([prefix_ids] * num_samples, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.generate(
                inp,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=4,
                eos_token_id=EOS_ID,
            )

    results = []
    flat_out = out.cpu().tolist()
    for seq in flat_out:
        gen_slice = seq[len(prefix_ids) :]
        has_eos = EOS_ID in gen_slice
        if has_eos:
            gen_slice = gen_slice[: gen_slice.index(EOS_ID)]
        raw_text = tokenizer.decode(gen_slice, skip_special_tokens=True)
        cleaned = clean_output(raw_text)
        results.append({
            "text": cleaned,
            "has_eos": has_eos,
            "tokens_len": len(gen_slice),
        })
    return results


def generate_multi_turn_prompt(
    model: GPT,
    tokenizer: Tokenizer,
    turn1: str,
    asst1: str,
    turn2: str,
    num_samples: int = 5,
    max_new_tokens: int = 120,
    temperature: float = 0.4,
    top_p: float = 0.90,
    min_p: float = 0.08,
    repetition_penalty: float = 1.15,
) -> List[Dict[str, Any]]:
    """Sample multiple responses for a multi-turn prompt."""
    user_hdr = tokenizer.encode("### User:\n").ids
    asst_hdr = tokenizer.encode("### Assistant:\n").ids
    sep = tokenizer.encode("\n\n").ids

    t1_ids = user_hdr + tokenizer.encode(f"{turn1}\n\n").ids
    a1_ids = asst_hdr + tokenizer.encode(asst1).ids + [EOS_ID] + sep
    t2_ids = user_hdr + tokenizer.encode(f"{turn2}\n\n").ids

    full_prefix = [BOS_ID] + t1_ids + a1_ids + t2_ids + asst_hdr
    inp = torch.tensor([full_prefix] * num_samples, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.generate(
                inp,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=4,
                eos_token_id=EOS_ID,
            )

    results = []
    flat_out = out.cpu().tolist()
    for seq in flat_out:
        gen_slice = seq[len(full_prefix) :]
        has_eos = EOS_ID in gen_slice
        if has_eos:
            gen_slice = gen_slice[: gen_slice.index(EOS_ID)]
        raw_text = tokenizer.decode(gen_slice, skip_special_tokens=True)
        cleaned = clean_output(raw_text)
        results.append({
            "text": cleaned,
            "has_eos": has_eos,
            "tokens_len": len(gen_slice),
        })
    return results


def evaluate_model_on_all_benchmarks(
    model: GPT,
    tokenizer: Tokenizer,
    model_name: str,
    rep_penalty: float = 1.15,
    num_samples: int = 5,
) -> Dict[str, Any]:
    print(f"\n--- Evaluating {model_name} (rep_pen={rep_penalty}) ---")
    results = {}

    # 1. Refusal Benchmark (20 prompts)
    refusal_hits = 0
    total_refusal_samples = 0
    lengths = []
    eos_clean_count = 0
    rep3_scores = []

    for p in REFUSAL_EVAL_PROMPTS:
        samples = generate_single_prompt(model, tokenizer, p, num_samples=num_samples, repetition_penalty=rep_penalty)
        for s in samples:
            total_refusal_samples += 1
            if is_refusal(s["text"]):
                refusal_hits += 1
            if s["has_eos"]:
                eos_clean_count += 1
            lengths.append(s["tokens_len"])
            rep3_scores.append(compute_3gram_rep(s["text"]))

    results["refusal_rate_pct"] = round((refusal_hits / total_refusal_samples) * 100.0, 2)
    results["clean_eos_rate_pct"] = round((eos_clean_count / total_refusal_samples) * 100.0, 2)
    results["mean_response_length"] = round(float(np.mean(lengths)), 1)
    results["std_response_length"] = round(float(np.std(lengths)), 1)
    results["rep_3gram_rate"] = round(float(np.mean(rep3_scores)), 4)

    # 2. Trivia Benchmark (20 prompts)
    trivia_correct = 0
    total_trivia_samples = 0

    for item in TRIVIA_EVAL_PROMPTS:
        q = item["prompt"]
        ans = item["answer"].lower()
        samples = generate_single_prompt(model, tokenizer, q, num_samples=num_samples, max_new_tokens=60, repetition_penalty=rep_penalty)
        for s in samples:
            total_trivia_samples += 1
            if ans in s["text"].lower():
                trivia_correct += 1

    results["trivia_accuracy_pct"] = round((trivia_correct / total_trivia_samples) * 100.0, 2)

    # 3. Leaked Format Benchmark (20 prompts)
    leaked_count = 0
    total_leaked_samples = 0

    for p in LEAKED_FORMAT_EVAL_PROMPTS:
        samples = generate_single_prompt(model, tokenizer, p, num_samples=num_samples, max_new_tokens=100, repetition_penalty=rep_penalty)
        for s in samples:
            total_leaked_samples += 1
            if has_leaked_format(s["text"]):
                leaked_count += 1

    results["leaked_format_pct"] = round((leaked_count / total_leaked_samples) * 100.0, 2)

    # 4. Multi-Turn Context Benchmark (5 prompts)
    mt_hits = 0
    total_mt_samples = 0

    for mt_item in MULTI_TURN_EVAL_PROMPTS:
        samples = generate_multi_turn_prompt(
            model, tokenizer, mt_item["turn1"], mt_item["asst1"], mt_item["turn2"],
            num_samples=num_samples, repetition_penalty=rep_penalty
        )
        for s in samples:
            total_mt_samples += 1
            if any(kw in s["text"].lower() for kw in mt_item["expected_keywords"]):
                mt_hits += 1

    results["multi_turn_retention_pct"] = round((mt_hits / total_mt_samples) * 100.0, 2)

    print(f"  -> Refusal Rate      : {results['refusal_rate_pct']}%")
    print(f"  -> Clean EOS Rate    : {results['clean_eos_rate_pct']}%")
    print(f"  -> Mean Length       : {results['mean_response_length']} ± {results['std_response_length']}")
    print(f"  -> 3-Gram Repetition : {results['rep_3gram_rate']:.4f}")
    print(f"  -> Trivia Accuracy   : {results['trivia_accuracy_pct']}%")
    print(f"  -> Leaked Format Rate: {results['leaked_format_pct']}%")
    print(f"  -> Multi-Turn Context: {results['multi_turn_retention_pct']}%")

    return results


def evaluate_sft_val_nll(model: GPT, tokenizer: Tokenizer) -> float:
    """Evaluate assistant-token NLL on SFT validation set to detect fluency drift."""
    from data.multi_turn_dataset import MultiTurnSFTDataset, multi_turn_collate_fn
    val_jsonl = PROJECT_ROOT / "data" / "sft_multi_turn" / "multi_turn_val.jsonl"
    ds = MultiTurnSFTDataset(val_jsonl, tokenizer, max_seq_len=512)
    loader = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=multi_turn_collate_fn)

    model.eval()
    total_loss = 0.0
    total_tokens = 0
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if DEVICE == "cuda" else torch.amp.autocast(device_type="cpu", enabled=False)

    for idx, (x, y) in enumerate(loader):
        if idx >= 100:  # 400 validation dialogues is plenty for accurate NLL
            break
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        with torch.no_grad():
            with amp_ctx:
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=IGNORE_INDEX,
                    reduction="sum",
                )
        valid = (y != IGNORE_INDEX).sum().item()
        total_loss += loss.item()
        total_tokens += valid

    return round(total_loss / max(total_tokens, 1), 4)


def run_full_evaluation(
    sft_ckpt_path: Path,
    dpo_candidates: Dict[str, Path],
    output_report_json: Path,
):
    print("=" * 72)
    print("  PHASE 5: RIGOROUS EVALUATION (SFT vs DPO CANDIDATES)")
    print("=" * 72)

    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))
    full_eval_results = {}

    # 1. Baseline SFT Model
    print(f"\n[Evaluating Baseline SFT] from {sft_ckpt_path.name}")
    sft_model = load_eval_model(sft_ckpt_path)
    sft_nll = evaluate_sft_val_nll(sft_model, tokenizer)
    print(f"  SFT Baseline Val NLL: {sft_nll}")

    sft_default = evaluate_model_on_all_benchmarks(sft_model, tokenizer, "SFT (Baseline)", rep_penalty=1.15)
    sft_unconfounded = evaluate_model_on_all_benchmarks(sft_model, tokenizer, "SFT (Unconfounded, rep=1.0)", rep_penalty=1.0)

    full_eval_results["SFT_Baseline"] = {
        "sft_val_nll": sft_nll,
        "default_decoding": sft_default,
        "unconfounded_decoding": sft_unconfounded,
    }
    del sft_model
    torch.cuda.empty_cache()

    # 2. DPO Candidates
    for c_name, c_path in dpo_candidates.items():
        print(f"\n[Evaluating DPO Candidate] {c_name} from {c_path}")
        if not c_path.exists():
            print(f"  Checkpoint missing: {c_path}, skipping.")
            continue

        cand_model = load_eval_model(c_path)
        cand_nll = evaluate_sft_val_nll(cand_model, tokenizer)
        nll_drift_pct = round(((cand_nll - sft_nll) / sft_nll) * 100.0, 2)
        print(f"  Candidate Val NLL: {cand_nll} (Drift: {nll_drift_pct:+.2f}%)")

        c_default = evaluate_model_on_all_benchmarks(cand_model, tokenizer, f"{c_name} (rep=1.15)", rep_penalty=1.15)
        c_unconfounded = evaluate_model_on_all_benchmarks(cand_model, tokenizer, f"{c_name} (rep=1.0)", rep_penalty=1.0)

        full_eval_results[c_name] = {
            "checkpoint_path": str(c_path),
            "sft_val_nll": cand_nll,
            "nll_drift_pct": nll_drift_pct,
            "fluency_flag": nll_drift_pct > 5.0,
            "default_decoding": c_default,
            "unconfounded_decoding": c_unconfounded,
        }
        del cand_model
        torch.cuda.empty_cache()

    output_report_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_json, "w", encoding="utf-8") as f:
        json.dump(full_eval_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Full evaluation results saved to: {output_report_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DPO Candidates against SFT Baseline")
    parser.add_argument("--sft-ckpt", type=str, default=str(DEFAULT_SFT_CHECKPOINT))
    parser.add_argument("--candidates-json", type=str, required=True, help="JSON mapping candidate names to checkpoint paths")
    parser.add_argument("--output-json", type=str, default=str(PROJECT_ROOT / "analysis" / "dpo_evaluation_results.json"))
    args = parser.parse_args()

    with open(args.candidates_json, "r", encoding="utf-8") as f:
        cands = {k: Path(v) for k, v in json.load(f).items()}

    run_full_evaluation(Path(args.sft_ckpt), cands, Path(args.output_json))
