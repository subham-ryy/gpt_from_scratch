import json
from pathlib import Path

p = Path("D:/pytorch/gpt_pretraining/experiments/20261122_020931_gpt_dpo_suite/evaluation_results.json")
data = json.load(open(p, encoding="utf-8"))

for k, v in data.items():
    print(f"\n==================== {k} ====================")
    print("  SFT Val NLL:", v.get("sft_val_nll"))
    print("  NLL Drift %:", v.get("nll_drift_pct"))
    d = v["default_decoding"]
    print(f"  [Default rep=1.15]")
    print(f"    Refusal Rate      : {d['refusal_rate_pct']}%")
    print(f"    3-Gram Repetition : {d['rep_3gram_rate']:.4f}")
    print(f"    Clean EOS Rate    : {d['clean_eos_rate_pct']}%")
    print(f"    Trivia Accuracy   : {d['trivia_accuracy_pct']}%")
    print(f"    Leaked Format Rate: {d['leaked_format_pct']}%")
    print(f"    Multi-Turn Context: {d['multi_turn_retention_pct']}%")
    print(f"    Mean Length       : {d['mean_response_length']}")
    u = v["unconfounded_decoding"]
    print(f"  [Unconfounded rep=1.00]")
    print(f"    Refusal Rate      : {u['refusal_rate_pct']}%")
    print(f"    3-Gram Repetition : {u['rep_3gram_rate']:.4f}")
    print(f"    Clean EOS Rate    : {u['clean_eos_rate_pct']}%")
    print(f"    Trivia Accuracy   : {u['trivia_accuracy_pct']}%")
    print(f"    Leaked Format Rate: {u['leaked_format_pct']}%")
    print(f"    Multi-Turn Context: {u['multi_turn_retention_pct']}%")
    print(f"    Mean Length       : {u['mean_response_length']}")
