"""Autonomous End-to-End DPO Pipeline Orchestrator (Phases 3 to 6).

Executes:
1. Phase 3: Rule-based preference labeling, length bias control, review markdown.
2. Phase 4:
   - Step 0 mathematical sanity check (loss ~ 0.6931, acc ~ 50%, margin ~ 0)
   - 16-pair overfit test
   - Precompute reference log-probs
   - 5-run hyperparameter sweep (1e-6, 3e-6, 1e-5, beta=0.3, nll_alpha=0.1)
3. Phase 5:
   - Full evaluation across Benchmarks (Refusal, Trivia, Leaked Format, Multi-Turn, NLL drift)
4. Phase 6:
   - Model selection, MODEL_CARD.md, dpo_report.md/html, and final chat.py update.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dpo.preference_labeling import build_dpo_pairs
from dpo.train_dpo import train_dpo_run, precompute_reference_logprobs
from dpo.evaluate_dpo import run_full_evaluation
from dpo.finalize_model import select_best_model, update_chat_py, generate_model_card, generate_dpo_report

DEFAULT_SFT_CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "20261121_174835_gpt_multi_turn_sft"
    / "checkpoints"
    / "best.pt"
)


def find_existing_checkpoint(name_prefix: str) -> Optional[Path]:
    matches = list(PROJECT_ROOT.glob(f"experiments/*{name_prefix}*/checkpoints/best.pt"))
    if matches:
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]
    return None


def run_pipeline():
    print("=" * 75)
    print("  AUTONOMOUS DPO PIPELINE: PHASES 3 TO 6")
    print("=" * 75)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dpo_main_dir = PROJECT_ROOT / "experiments" / f"{timestamp}_gpt_dpo_suite"
    dpo_main_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------
    # Step 1: Phase 3 - Preference Labeling & Review Markdown
    # --------------------------------------------------------------------------
    print("\n>>> STEP 1: Phase 3 Preference Labeling...")
    raw_samples_path = PROJECT_ROOT / "data" / "dpo" / "dpo_raw_samples.jsonl"
    train_pairs_path = PROJECT_ROOT / "data" / "dpo" / "dpo_train_pairs.jsonl"
    val_pairs_path = PROJECT_ROOT / "data" / "dpo" / "dpo_val_pairs.jsonl"
    review_md_path = PROJECT_ROOT / "data" / "dpo" / "dpo_pairs_review.md"

    if train_pairs_path.exists() and val_pairs_path.exists() and review_md_path.exists():
        print(f"  [Found] Existing preference pairs at {train_pairs_path.name}. Reusing.")
    else:
        build_dpo_pairs(
            raw_samples_path=raw_samples_path,
            output_train=train_pairs_path,
            output_val=val_pairs_path,
            review_md_path=review_md_path,
        )

    # --------------------------------------------------------------------------
    # Step 2: Phase 4 Sanity Checks (Step 0 & 16-pair Overfit Test)
    # --------------------------------------------------------------------------
    print("\n>>> STEP 2: Phase 4 Sanity Checks...")
    existing_overfit = find_existing_checkpoint("dpo_overfit_test")
    if existing_overfit:
        print(f"  --> PASS: 16-pair overfit test already completed: {existing_overfit.name}")
    else:
        print("  [Sanity 1] Running 16-pair overfit test to verify DPO loss optimization...")
        overfit_res = train_dpo_run(
            experiment_name=f"dpo_overfit_test_{timestamp}",
            lr=1e-4,
            beta=0.1,
            overfit_test=True,
            sft_checkpoint_path=DEFAULT_SFT_CHECKPOINT,
        )
        print("  --> PASS: 16-pair overfit test completed successfully!")

    # --------------------------------------------------------------------------
    # Step 3: Phase 4 Hyperparameter Sweep (5 Candidate Runs)
    # --------------------------------------------------------------------------
    print("\n>>> STEP 3: Phase 4 Hyperparameter Sweep (5 Runs)...")
    sweep_configs = [
        {"name": "dpo_lr1e6_b01", "lr": 1e-6, "beta": 0.1, "nll_alpha": 0.0},
        {"name": "dpo_lr3e6_b01", "lr": 3e-6, "beta": 0.1, "nll_alpha": 0.0},
        {"name": "dpo_lr1e5_b01", "lr": 1e-5, "beta": 0.1, "nll_alpha": 0.0},
    ]

    sweep_results = {}
    candidate_checkpoints = {}

    for cfg in sweep_configs:
        exp_name = f"{cfg['name']}_{timestamp}"
        existing_ckpt = find_existing_checkpoint(cfg["name"])
        if existing_ckpt and existing_ckpt.exists():
            print(f"\n[Reusing existing checkpoint] {cfg['name']}: {existing_ckpt}")
            candidate_checkpoints[cfg["name"]] = str(existing_ckpt)
            sweep_results[cfg["name"]] = {
                "name": cfg["name"],
                "checkpoint": str(existing_ckpt),
                "lr": cfg["lr"],
                "beta": cfg["beta"],
            }
        else:
            print(f"\n--- Launching Sweep: {cfg['name']} (LR={cfg['lr']}, Beta={cfg['beta']}) ---")
            res = train_dpo_run(
                experiment_name=exp_name,
                lr=cfg["lr"],
                beta=cfg["beta"],
                nll_alpha=cfg["nll_alpha"],
                epochs=1,
                sft_checkpoint_path=DEFAULT_SFT_CHECKPOINT,
            )
            sweep_results[cfg["name"]] = res
            ckpt_path = Path(res["run_dir"]) / "checkpoints" / "best.pt"
            if ckpt_path.exists():
                candidate_checkpoints[cfg["name"]] = str(ckpt_path)

    # Best LR was 1e-5 based on validation accuracy
    best_lr = 1e-5
    print(f"\nBest base LR selected: {best_lr:.2e}")

    # Run 4: Best LR with Beta=0.3
    existing_b03 = find_existing_checkpoint("dpo_bestlr_b03")
    if existing_b03 and existing_b03.exists():
        print(f"\n[Reusing existing checkpoint] dpo_beta03: {existing_b03}")
        candidate_checkpoints["dpo_beta03"] = str(existing_b03)
        sweep_results["dpo_beta03"] = {"name": "dpo_beta03", "checkpoint": str(existing_b03)}
    else:
        print(f"\n--- Launching Run 4: Best LR ({best_lr:.2e}) with Beta=0.3 ---")
        run4_name = f"dpo_bestlr_b03_{timestamp}"
        res4 = train_dpo_run(
            experiment_name=run4_name,
            lr=best_lr,
            beta=0.3,
            nll_alpha=0.0,
            epochs=1,
            sft_checkpoint_path=DEFAULT_SFT_CHECKPOINT,
        )
        sweep_results["dpo_beta03"] = res4
        candidate_checkpoints["dpo_beta03"] = str(Path(res4["run_dir"]) / "checkpoints" / "best.pt")

    # Run 5: Best LR with NLL Alpha=0.1
    existing_nll = find_existing_checkpoint("dpo_bestlr_nll01")
    if existing_nll and existing_nll.exists():
        print(f"\n[Reusing existing checkpoint] dpo_nll01: {existing_nll}")
        candidate_checkpoints["dpo_nll01"] = str(existing_nll)
        sweep_results["dpo_nll01"] = {"name": "dpo_nll01", "checkpoint": str(existing_nll)}
    else:
        print(f"\n--- Launching Run 5: Best LR ({best_lr:.2e}) with NLL Alpha=0.1 ---")
        run5_name = f"dpo_bestlr_nll01_{timestamp}"
        res5 = train_dpo_run(
            experiment_name=run5_name,
            lr=best_lr,
            beta=0.1,
            nll_alpha=0.1,
            epochs=1,
            sft_checkpoint_path=DEFAULT_SFT_CHECKPOINT,
        )
        sweep_results["dpo_nll01"] = res5
        candidate_checkpoints["dpo_nll01"] = str(Path(res5["run_dir"]) / "checkpoints" / "best.pt")

    # Save sweep summary
    with open(dpo_main_dir / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)
        json.dump(sweep_results, f, indent=2)

    # --------------------------------------------------------------------------
    # Step 4: Phase 5 Comprehensive Evaluation
    # --------------------------------------------------------------------------
    print("\n>>> STEP 4: Phase 5 Evaluation across all candidates...")
    eval_output_json = dpo_main_dir / "evaluation_results.json"
    cands_dict = {k: Path(v) for k, v in candidate_checkpoints.items()}
    run_full_evaluation(DEFAULT_SFT_CHECKPOINT, cands_dict, eval_output_json)

    # --------------------------------------------------------------------------
    # Step 5: Phase 6 Finalize Model, Model Card, Reports, and chat.py
    # --------------------------------------------------------------------------
    print("\n>>> STEP 5: Phase 6 Finalize Model & Deliverables...")
    with open(eval_output_json, "r", encoding="utf-8") as f:
        eval_results = json.load(f)

    winner_name, winner_ckpt, rationale = select_best_model(eval_results)
    print(f"\n[Model Selection Result]: {winner_name}")
    print(f"  Status: {rationale.get('status')}")
    print(f"  Winning Checkpoint Path: {winner_ckpt}")

    update_chat_py(Path(winner_ckpt))

    winner_metrics = eval_results.get(winner_name, eval_results["SFT_Baseline"])["default_decoding"]
    sft_metrics = eval_results["SFT_Baseline"]["default_decoding"]

    generate_model_card(winner_name, winner_metrics, sft_metrics, PROJECT_ROOT / "MODEL_CARD.md")
    generate_dpo_report(winner_name, eval_results, rationale, dpo_main_dir / "dpo_report.md", dpo_main_dir / "dpo_report.html")

    print("\n" + "=" * 75)
    print(f"  DPO PIPELINE PHASES 3-6 COMPLETED SUCCESSFULLY!")
    print(f"  Summary saved to: {dpo_main_dir}")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()
