import json

with open(r"D:\pytorch\gpt_pretraining\data\dpo\dpo_raw_samples.jsonl", "r", encoding="utf-8") as f:
    for i in range(2):
        line = f.readline()
        if not line:
            break
        d = json.loads(line)
        print(f"=== Prompt {i+1} [{d['category']}] ===")
        print("User:", d["prompt_messages"][-1]["content"])
        for s_i, s in enumerate(d["samples"]):
            print(f"  Sample {s_i+1} (eos={s['has_eos']}, len={s['raw_tokens_len']}): {s['response_text'][:110]}...")
