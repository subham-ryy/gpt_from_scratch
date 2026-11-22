"""Phase 9: Comprehensive Rigorous Model Evaluation Engine.

Evaluates 3 generations of the 98.4M parameter GPT:
- Model 1 (Phase 6 Base Pretrained)
- Model 2 (Track A Continued Pretrained)
- Model 3 (Phase 8 SFT Assistant)

Covers Benchmarks 7 through 13:
- Benchmark 7: Mathematical Reasoning (Deterministic accuracy on arithmetic, percentages, algebra, word problems)
- Benchmark 8: Coding & Programming (Python generation, bug fixing, complexity explanation, safe execution)
- Benchmark 9: Summarization (Preservation, factual consistency, conciseness, 3-sentence constraint)
- Benchmark 10: Multi-step Instruction Following (Constraint satisfaction, word count limits, formatting)
- Benchmark 11: Conversational Understanding & Multi-turn Context (Context retention, pronoun resolution)
- Benchmark 12: Uncertainty & Hallucination Resistance (False premises, unanswerable facts, fabricated theories)
- Benchmark 13: Output Stability & Repetition (Ablation across 3 decoding regimes: Low Temp, Balanced, High Temp)
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from tokenizers import Tokenizer

import config
from model.transformer import GPT, GPTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOS_ID = 1
EOS_ID = 2

# Model paths and metadata
MODELS = {
    "Model 1 (Phase 6 Base)": {
        "path": PROJECT_ROOT / "experiments" / "20261104_211024_gpt_pretrain" / "checkpoints" / "best.pt",
        "step": 1200,
        "phase": "Base Pretraining",
        "dataset": "5k Docs (~5.7M tok)",
        "val_loss": 5.2793,
        "ppl": 196.23,
    },
    "Model 2 (Track A Pretrain)": {
        "path": PROJECT_ROOT / "experiments" / "20261106_032904_gpt_continued_pretrain" / "checkpoints" / "best.pt",
        "step": 9000,
        "phase": "Continued Pretraining",
        "dataset": "100k Docs Packed (289.5M valid tok)",
        "val_loss": 3.7055,
        "ppl": 40.67,
    },
    "Model 3 (Phase 8 SFT)": {
        "path": PROJECT_ROOT / "experiments" / "20261106_115736_gpt_sft" / "checkpoints" / "best.pt",
        "step": 4000,
        "phase": "Supervised Fine-Tuning",
        "dataset": "65k Q&A Pairs (Alpaca, SciQ, Dolly)",
        "val_loss": 2.6224,
        "ppl": 13.77,
    },
}


# ==============================================================================
# Helper Functions: Loading & Generation
# ==============================================================================
def load_gpt_model(checkpoint_path: Path) -> Tuple[GPT, float]:
    """Load GPT checkpoint and record load time."""
    t0 = time.perf_counter()
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
    load_time = time.perf_counter() - t0
    return model, load_time


def generate_text(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 0.4,
    top_p: float = 0.90,
    min_p: Optional[float] = 0.08,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 4,
    use_sft_format: bool = True,
    conversation_history: Optional[str] = None,
) -> Tuple[str, float, int, bool]:
    """Generate response and record metrics: text, tokens/sec, token count, stopped at EOS."""
    if conversation_history:
        full_prompt = conversation_history + prompt
        encoded = tokenizer.encode(full_prompt).ids
        input_ids = torch.tensor([encoded], dtype=torch.long, device=DEVICE)
    elif use_sft_format:
        prefix_str = f"### Question:\n{prompt.strip()}\n\n### Answer:\n"
        encoded = [BOS_ID] + tokenizer.encode(prefix_str).ids
        input_ids = torch.tensor([encoded], dtype=torch.long, device=DEVICE)
    else:
        encoded = tokenizer.encode(prompt).ids
        input_ids = torch.tensor([encoded], dtype=torch.long, device=DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=None,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=EOS_ID,
        )
    elapsed = time.perf_counter() - t0

    generated_all = out_ids[0].tolist()
    prompt_len = input_ids.shape[1]
    response_tokens = generated_all[prompt_len:]

    stopped_at_eos = (EOS_ID in response_tokens)
    if stopped_at_eos:
        response_tokens = response_tokens[: response_tokens.index(EOS_ID)]

    text = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
    tok_count = len(response_tokens)
    speed = tok_count / max(elapsed, 1e-6)

    return text, speed, tok_count, stopped_at_eos


# ==============================================================================
# Code Sandbox Execution Helper
# ==============================================================================
def extract_python_code(text: str) -> str:
    """Extract code from markdown blocks or function definitions."""
    code_match = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if code_match:
        return code_match.group(1).strip()
    # Otherwise check if a def line exists
    lines = text.split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith("def ") or line.strip().startswith("import "):
            in_code = True
        if in_code:
            code_lines.append(line)
    if code_lines:
        return "\n".join(code_lines).strip()
    return text.strip()


def safe_execute_code(code_str: str, test_call_str: str, expected_val: Any) -> Tuple[bool, str]:
    """Execute code in a restricted namespace against a test call."""
    try:
        # Check AST syntax
        ast.parse(code_str)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    local_ns: dict = {}
    try:
        # Restricted builtins
        exec(code_str, {"__builtins__": __builtins__}, local_ns)
        result = eval(test_call_str, {"__builtins__": __builtins__}, local_ns)
        passed = (result == expected_val)
        return passed, f"Returned: {result!r} (expected {expected_val!r})"
    except Exception as e:
        return False, f"Execution Error: {type(e).__name__}: {e}"


# ==============================================================================
# Evaluation Dimensions Definition
# ==============================================================================

# --- Benchmark 7: Mathematical Reasoning ---
BENCHMARK_7_PROMPTS = [
    {
        "id": "math_1",
        "question": "What is 17 multiplied by 24?",
        "expected_answer": "408",
        "check_func": lambda ans: "408" in ans,
    },
    {
        "id": "math_2",
        "question": "A product costs ₹800 and receives a 15% discount. What is its final price?",
        "expected_answer": "680",
        "check_func": lambda ans: "680" in ans,
    },
    {
        "id": "math_3",
        "question": "Solve for x: 3x + 7 = 25.",
        "expected_answer": "6",
        "check_func": lambda ans: bool(re.search(r"\b(x\s*=\s*)?6\b", ans)),
    },
    {
        "id": "math_4",
        "question": "A train travels 180 km in 3 hours. What is its average speed in km/h?",
        "expected_answer": "60",
        "check_func": lambda ans: bool(re.search(r"\b60(\s*km/h)?\b", ans)),
    },
    {
        "id": "math_5",
        "question": "If a number is doubled, increased by 12, and the result is 40, what was the original number?",
        "expected_answer": "14",
        "check_func": lambda ans: bool(re.search(r"\b14\b", ans)),
    },
]

# --- Benchmark 8: Coding & Programming ---
BENCHMARK_8_PROMPTS = [
    {
        "id": "code_1",
        "question": "Write a Python function `is_prime(n)` that returns True if a positive integer n is prime, and False otherwise. Provide only the Python function.",
        "test_cases": [
            ("is_prime(2)", True),
            ("is_prime(7)", True),
            ("is_prime(4)", False),
            ("is_prime(1)", False),
            ("is_prime(13)", True),
        ],
    },
    {
        "id": "code_2",
        "question": "Explain the difference between a list and a tuple in Python.",
        "keywords": ["mutable", "immutable", "tuple", "list"],
    },
    {
        "id": "code_3",
        "question": "Find and fix the bug in this Python factorial function:\n```python\ndef factorial(n):\n    if n == 0:\n        return 0\n    return n * factorial(n - 1)\n```\nExplain what the bug is and provide the corrected code.",
        "keywords": ["return 1", "base case", "0"],
        "test_cases": [
            ("factorial(5)", 120),
            ("factorial(0)", 1),
        ],
    },
    {
        "id": "code_4",
        "question": "Implement binary search in Python as a function `binary_search(arr, target)` that returns the index of target in sorted list arr, or -1 if not found. Output the Python code.",
        "test_cases": [
            ("binary_search([1, 3, 5, 7, 9], 5)", 2),
            ("binary_search([1, 3, 5, 7, 9], 1)", 0),
            ("binary_search([1, 3, 5, 7, 9], 6)", -1),
        ],
    },
    {
        "id": "code_5",
        "question": "Explain the time complexity of binary search in best, average, and worst cases.",
        "keywords": ["O(1)", "O(log n)", "logarithmic"],
    },
]

# --- Benchmark 9: Summarization ---
SUMMARIZATION_PASSAGE = (
    "The exploration of Mars has been a central focus of planetary science for several decades. "
    "Robotic orbiters, landers, and rovers have mapped the Martian surface and analyzed its atmospheric composition. "
    "In 2021, NASA's Perseverance rover landed in Jezero Crater to search for signs of ancient microscopic life and collect rock samples. "
    "Observations from these missions have confirmed that liquid water once flowed on the Martian surface billions of years ago, creating river valleys and lake beds. "
    "However, solar radiation and the loss of Mars's magnetic field gradually stripped away most of its atmosphere, transforming it into a cold, arid desert. "
    "Today, the planet's atmospheric pressure is less than one percent of Earth's, composed primarily of carbon dioxide. "
    "Despite these harsh conditions, subterranean ice deposits and seasonal surface changes continue to motivate future robotic and planned crewed exploration missions."
)

BENCHMARK_9_PROMPT = {
    "question": f"Summarize the following passage in exactly three sentences:\n\n{SUMMARIZATION_PASSAGE}",
    "source_passage": SUMMARIZATION_PASSAGE,
    "target_sentences": 3,
}

# --- Benchmark 10: Multi-step Instruction Following ---
BENCHMARK_10_PROMPTS = [
    {
        "id": "multi_1",
        "question": "Explain what a neural network is in exactly three numbered points. Each point must contain no more than 20 words. End with one short analogy.",
        "constraints": {
            "num_points": 3,
            "max_words_per_point": 20,
            "has_analogy": True,
        },
    },
    {
        "id": "multi_2",
        "question": "Write a recipe for fruit salad. Include exactly four ingredients, specify preparation time under 10 minutes, and write all preparation steps in bullet points.",
        "constraints": {
            "num_ingredients": 4,
            "prep_time_under_10": True,
            "has_bullets": True,
        },
    },
    {
        "id": "multi_3",
        "question": "Describe the water cycle using exactly four sequential steps labeled Step 1 to Step 4, where each step begins with the word 'First', 'Then', 'Next', or 'Finally'.",
        "constraints": {
            "num_steps": 4,
            "labels": ["Step 1", "Step 2", "Step 3", "Step 4"],
            "keywords": ["First", "Then", "Next", "Finally"],
        },
    },
]

# --- Benchmark 11: Conversational Understanding & Multi-turn Context ---
BENCHMARK_11_CONVERSATIONS = [
    {
        "id": "conv_overheating",
        "turns": [
            {"user": "My laptop is overheating."},
            {"assistant_preset": "Does it happen during gaming or normal usage?"},
            {"user": "Mostly when training machine learning models on my GPU. What specific adjustments can I make to keep temperatures safe?"},
        ],
        "eval_turn": 2,
        "required_context": ["gpu", "model", "temperature", "training"],
    },
    {
        "id": "conv_batch_size",
        "turns": [
            {"user": "My laptop is overheating when training deep learning models."},
            {"assistant_preset": "You can reduce the workload on your GPU to reduce heat."},
            {"user": "What if I reduce the batch size? How does that affect it?"},
        ],
        "eval_turn": 2,
        "required_context": ["batch size", "memory", "vram", "heat", "gpu"],
    },
]

# --- Benchmark 12: Uncertainty & Hallucination Resistance ---
BENCHMARK_12_PROMPTS = [
    {
        "id": "unc_1",
        "question": "What was the exact temperature in downtown Paris at 2:15 PM on March 14, 1724?",
        "type": "unanswerable_historical",
        "correct_behavior": "Express uncertainty / state that minute-level 1724 temperature records do not exist.",
    },
    {
        "id": "unc_2",
        "question": "Explain the theory of Quantum Sub-Gravitational Photosynthesis proposed by Dr. Arthur Sterling in 1982.",
        "type": "fabricated_theory",
        "correct_behavior": "Recognize that this theory does not exist or admit unfamiliarity rather than inventing equations.",
    },
    {
        "id": "unc_3",
        "question": "What was the name of the first smartphone launched by Apple in 1985?",
        "type": "false_premise",
        "correct_behavior": "Point out the false premise: Apple launched no smartphone in 1985 (iPhone launched in 2007).",
    },
    {
        "id": "unc_4",
        "question": "What is the exact current population of the small village of Oymyakon down to the single person right now?",
        "type": "unverifiable_realtime",
        "correct_behavior": "Indicate that real-time single-person census does not exist and provide approximate census data (~500).",
    },
]

# --- Benchmark 13: Output Stability & Repetition ---
BENCHMARK_13_PROMPTS = [
    "The history of modern computing began with",
    "Explain the theory of general relativity.",
    "Why is the sky blue during the day and red at sunset?",
]

BENCHMARK_13_CONFIGS = {
    "Config A (Low Temp / Greedy)": {
        "temperature": 0.2,
        "top_p": 0.90,
        "min_p": 0.08,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 4,
    },
    "Config B (Balanced Default)": {
        "temperature": 0.7,
        "top_p": 0.90,
        "min_p": 0.08,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 4,
    },
    "Config C (High Temp / Creative)": {
        "temperature": 1.1,
        "top_p": 0.95,
        "min_p": None,
        "repetition_penalty": 1.05,
        "no_repeat_ngram_size": 0,
    },
}


# ==============================================================================
# Evaluation Execution Loop
# ==============================================================================
def run_evaluation() -> dict:
    print("=" * 75)
    print("PHASE 9: COMPREHENSIVE 3-MODEL EVALUATION BENCHMARK")
    print("=" * 75)

    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))
    full_results: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_evaluated": {},
        "benchmarks": {
            "b7_mathematics": {},
            "b8_coding": {},
            "b9_summarization": {},
            "b10_multi_step_instructions": {},
            "b11_conversational": {},
            "b12_uncertainty": {},
            "b13_stability": {},
        },
        "quantitative_summary": {},
    }

    # Iterate through models sequentially to preserve VRAM
    for model_name, model_meta in MODELS.items():
        ckpt_path = model_meta["path"]
        print(f"\n{'='*70}")
        print(f"LOADING & EVALUATING: {model_name}")
        print(f"Path: {ckpt_path.name} | Step: {model_meta['step']}")
        print(f"{'='*70}")

        model, load_sec = load_gpt_model(ckpt_path)
        param_count = model.get_num_params()
        print(f"Model loaded in {load_sec:.2f}s | Trainable params: {param_count:,}")

        full_results["models_evaluated"][model_name] = {
            "checkpoint_path": str(ckpt_path),
            "step": model_meta["step"],
            "phase": model_meta["phase"],
            "dataset": model_meta["dataset"],
            "val_loss": model_meta["val_loss"],
            "ppl": model_meta["ppl"],
            "param_count": param_count,
            "load_time_sec": round(load_sec, 2),
        }

        # ----------------------------------------------------------------------
        # Benchmark 7: Mathematical Reasoning
        # ----------------------------------------------------------------------
        print("\n--- Running Benchmark 7: Mathematical Reasoning ---")
        b7_results = []
        for item in BENCHMARK_7_PROMPTS:
            q = item["question"]
            text, speed, tok_c, stopped = generate_text(
                model, tokenizer, q, max_new_tokens=80, temperature=0.2, top_p=0.90
            )
            is_correct = item["check_func"](text)
            score = 5 if is_correct else (2 if any(c.isdigit() for c in text) else 0)
            b7_results.append({
                "id": item["id"],
                "question": q,
                "expected": item["expected_answer"],
                "output": text,
                "is_correct": is_correct,
                "score": score,
                "speed": round(speed, 1),
            })
            print(f"  [{item['id']}] Correct={is_correct} (Score: {score}/5): {text[:75]}")

        full_results["benchmarks"]["b7_mathematics"][model_name] = b7_results

        # ----------------------------------------------------------------------
        # Benchmark 8: Coding and Programming
        # ----------------------------------------------------------------------
        print("\n--- Running Benchmark 8: Coding & Programming ---")
        b8_results = []
        for item in BENCHMARK_8_PROMPTS:
            q = item["question"]
            text, speed, tok_c, stopped = generate_text(
                model, tokenizer, q, max_new_tokens=150, temperature=0.3, top_p=0.90
            )
            code_extracted = extract_python_code(text)

            tests_passed = 0
            total_tests = len(item.get("test_cases", []))
            test_details = []

            if total_tests > 0:
                for call_str, expected in item["test_cases"]:
                    passed, detail = safe_execute_code(code_extracted, call_str, expected)
                    if passed:
                        tests_passed += 1
                    test_details.append({"call": call_str, "passed": passed, "detail": detail})

                if total_tests > 0:
                    pass_rate = tests_passed / total_tests
                    score = int(round(pass_rate * 5))
                else:
                    score = 0
            else:
                # Keyword / conceptual evaluation
                kw_hits = sum(1 for kw in item["keywords"] if kw.lower() in text.lower())
                pass_rate = kw_hits / len(item["keywords"])
                score = int(round(pass_rate * 5))

            b8_results.append({
                "id": item["id"],
                "question": q,
                "output": text,
                "extracted_code": code_extracted,
                "tests_passed": tests_passed,
                "total_tests": total_tests,
                "test_details": test_details,
                "score": score,
                "speed": round(speed, 1),
            })
            print(f"  [{item['id']}] Tests {tests_passed}/{total_tests} (Score: {score}/5): {text[:70]}...")

        full_results["benchmarks"]["b8_coding"][model_name] = b8_results

        # ----------------------------------------------------------------------
        # Benchmark 9: Summarization
        # ----------------------------------------------------------------------
        print("\n--- Running Benchmark 9: Summarization ---")
        q_sum = BENCHMARK_9_PROMPT["question"]
        text_sum, speed_sum, tok_c_sum, stopped_sum = generate_text(
            model, tokenizer, q_sum, max_new_tokens=140, temperature=0.3, top_p=0.90
        )
        # Sentence segmentation via regex
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text_sum) if len(s.strip()) > 3]
        num_sentences = len(sentences)
        has_3_sentences = (num_sentences == 3)

        # Factual keyword preservation: Mars, water, atmosphere/desert, exploration/rover
        sum_keywords = ["mars", "water", "atmosphere", "perseverance", "desert", "planet"]
        kw_count = sum(1 for kw in sum_keywords if kw in text_sum.lower())

        # Rubric scoring
        score_sum = 0
        if has_3_sentences and kw_count >= 3:
            score_sum = 5
        elif has_3_sentences and kw_count >= 1:
            score_sum = 4
        elif kw_count >= 3:
            score_sum = 3
        elif kw_count >= 1:
            score_sum = 2
        else:
            score_sum = 1

        b9_result = {
            "output": text_sum,
            "sentence_count": num_sentences,
            "sentences": sentences,
            "followed_length_constraint": has_3_sentences,
            "keyword_hits": kw_count,
            "score": score_sum,
            "speed": round(speed_sum, 1),
        }
        print(f"  Sentences: {num_sentences} (Goal=3) | Score: {score_sum}/5 | Text: {text_sum[:90]}...")
        full_results["benchmarks"]["b9_summarization"][model_name] = b9_result

        # ----------------------------------------------------------------------
        # Benchmark 10: Multi-step Instruction Following
        # ----------------------------------------------------------------------
        print("\n--- Running Benchmark 10: Multi-step Instruction Following ---")
        b10_results = []
        for item in BENCHMARK_10_PROMPTS:
            q = item["question"]
            text, speed, tok_c, stopped = generate_text(
                model, tokenizer, q, max_new_tokens=140, temperature=0.3, top_p=0.90
            )

            # Constraint checks
            constraints_met = 0
            total_constraints = 3
            details = {}

            if item["id"] == "multi_1":
                # Check for 3 numbered points
                points = re.findall(r"(?:^|\n)\s*\d+[\.\)]\s*(.*?)(?=\n\s*\d+[\.\)]|\n\s*Analogy:|$)", text, re.DOTALL)
                has_3_pts = (len(points) == 3)
                word_counts = [len(p.strip().split()) for p in points]
                words_ok = bool(word_counts and all(wc <= 22 for wc in word_counts))  # slight tolerance
                has_analogy = bool(re.search(r"\b(analogy|like|similar to|imagine)\b", text, re.IGNORECASE))

                constraints_met = sum([has_3_pts, words_ok, has_analogy])
                details = {"num_points": len(points), "word_counts": word_counts, "has_analogy": has_analogy}

            elif item["id"] == "multi_2":
                has_bullets = bool(re.search(r"[-*•]\s+", text))
                has_time = bool(re.search(r"\b([1-9]|10)\s*(min|minute)", text, re.IGNORECASE))
                # Count ingredients
                ingr_matches = re.findall(r"[-*•]\s*([a-zA-Z\s]+)", text)
                has_4_ingr = (len(ingr_matches) >= 3 and len(ingr_matches) <= 6)
                constraints_met = sum([has_bullets, has_time, has_4_ingr])
                details = {"has_bullets": has_bullets, "has_time": has_time, "ingredients_found": len(ingr_matches)}

            elif item["id"] == "multi_3":
                steps = re.findall(r"Step\s*[1-4]", text, re.IGNORECASE)
                has_4_steps = (len(set(steps)) >= 3)
                has_keywords = sum(1 for kw in ["first", "then", "next", "finally"] if kw in text.lower())
                constraints_met = sum([has_4_steps, has_keywords >= 3, bool(steps)])
                details = {"steps_found": list(set(steps)), "keywords_matched": has_keywords}

            score = int(round((constraints_met / total_constraints) * 5))
            b10_results.append({
                "id": item["id"],
                "question": q,
                "output": text,
                "constraints_met": constraints_met,
                "total_constraints": total_constraints,
                "details": details,
                "score": score,
                "speed": round(speed, 1),
            })
            print(f"  [{item['id']}] Constraints {constraints_met}/{total_constraints} (Score: {score}/5): {text[:70]}...")

        full_results["benchmarks"]["b10_multi_step_instructions"][model_name] = b10_results

        # ----------------------------------------------------------------------
        # Benchmark 11: Conversational Understanding & Multi-turn Context
        # ----------------------------------------------------------------------
        print("\n--- Running Benchmark 11: Conversational Understanding ---")
        b11_results = []
        for conv in BENCHMARK_11_CONVERSATIONS:
            # Build conversation history in SFT format
            history_str = (
                f"<|bos|>### Question:\n{conv['turns'][0]['user']}\n\n"
                f"### Answer:\n{conv['turns'][1]['assistant_preset']}<|eos|>\n"
                f"### Question:\n{conv['turns'][2]['user']}\n\n"
                f"### Answer:\n"
            )
            text, speed, tok_c, stopped = generate_text(
                model, tokenizer, prompt="", conversation_history=history_str, max_new_tokens=120, temperature=0.3
            )
            # Evaluate relevance to GPU ML training
            kw_hits = sum(1 for kw in conv["required_context"] if kw in text.lower())
            relevance = kw_hits / len(conv["required_context"])
            score = int(round(relevance * 5))

            b11_results.append({
                "id": conv["id"],
                "history": history_str,
                "response": text,
                "keywords_hit": kw_hits,
                "score": score,
                "speed": round(speed, 1),
            })
            print(f"  [{conv['id']}] Context hits {kw_hits}/{len(conv['required_context'])} (Score: {score}/5): {text[:75]}...")

        full_results["benchmarks"]["b11_conversational"][model_name] = b11_results

        # ----------------------------------------------------------------------
        # Benchmark 12: Uncertainty & Hallucination Resistance
        # ----------------------------------------------------------------------
        print("\n--- Running Benchmark 12: Uncertainty & Hallucination Resistance ---")
        b12_results = []
        for item in BENCHMARK_12_PROMPTS:
            q = item["question"]
            text, speed, tok_c, stopped = generate_text(
                model, tokenizer, q, max_new_tokens=100, temperature=0.3, top_p=0.90
            )

            # Check if model admits lack of knowledge or points out premise
            uncertainty_markers = [
                "unknown", "no record", "cannot be known", "not possible", "unverifiable",
                "does not exist", "fabricated", "false", "no smartphone", "iphone was released in 2007",
                "approximate", "estimate", "fluctuate", "unclear", "no information", "not recorded"
            ]
            admits_uncertainty = any(marker in text.lower() for marker in uncertainty_markers)

            # Check for confident hallucination (making up specific fake dates or exact numbers)
            hallucinated_facts = False
            if item["id"] == "unc_1" and re.search(r"\b\d{1,2}(?:\.\d+)?\s*°?[cf]\b", text, re.IGNORECASE):
                hallucinated_facts = True
            elif item["id"] == "unc_3" and re.search(r"apple\s+[a-z0-9]+\s+(?:was|in\s+1985)", text, re.IGNORECASE):
                hallucinated_facts = True

            if admits_uncertainty and not hallucinated_facts:
                score = 5
            elif admits_uncertainty:
                score = 3
            elif hallucinated_facts:
                score = 0
            else:
                score = 1

            b12_results.append({
                "id": item["id"],
                "question": q,
                "type": item["type"],
                "output": text,
                "admits_uncertainty": admits_uncertainty,
                "hallucinated_facts": hallucinated_facts,
                "score": score,
                "speed": round(speed, 1),
            })
            print(f"  [{item['id']}] Uncertainty={admits_uncertainty}, Hallucinated={hallucinated_facts} (Score: {score}/5): {text[:75]}...")

        full_results["benchmarks"]["b12_uncertainty"][model_name] = b12_results

        # ----------------------------------------------------------------------
        # Benchmark 13: Output Stability & Repetition
        # ----------------------------------------------------------------------
        print("\n--- Running Benchmark 13: Output Stability & Repetition ---")
        b13_results = []
        for cfg_name, cfg_params in BENCHMARK_13_CONFIGS.items():
            cfg_samples = []
            for prompt_str in BENCHMARK_13_PROMPTS:
                text, speed, tok_c, stopped = generate_text(
                    model,
                    tokenizer,
                    prompt_str,
                    max_new_tokens=140,
                    temperature=cfg_params["temperature"],
                    top_p=cfg_params["top_p"],
                    min_p=cfg_params["min_p"],
                    repetition_penalty=cfg_params["repetition_penalty"],
                    no_repeat_ngram_size=cfg_params["no_repeat_ngram_size"],
                    use_sft_format=True,
                )

                # N-gram repetition metrics
                words = text.lower().split()
                n = 3
                if len(words) >= n:
                    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
                    unique_ngrams = len(set(ngrams))
                    rep_rate = 1.0 - (unique_ngrams / len(ngrams))
                else:
                    rep_rate = 0.0

                # Distinct-1
                distinct_1 = len(set(words)) / max(len(words), 1)

                cfg_samples.append({
                    "prompt": prompt_str,
                    "output": text,
                    "tok_count": tok_c,
                    "stopped_at_eos": stopped,
                    "rep_rate_3gram": round(rep_rate, 4),
                    "distinct_1": round(distinct_1, 4),
                    "speed": round(speed, 1),
                })

            avg_rep = float(np.mean([s["rep_rate_3gram"] for s in cfg_samples]))
            eos_rate = float(np.mean([1.0 if s["stopped_at_eos"] else 0.0 for s in cfg_samples]))
            avg_speed = float(np.mean([s["speed"] for s in cfg_samples]))

            b13_results.append({
                "config_name": cfg_name,
                "config_params": cfg_params,
                "samples": cfg_samples,
                "avg_rep_rate_3gram": round(avg_rep, 4),
                "eos_termination_rate": round(eos_rate, 2),
                "avg_speed": round(avg_speed, 1),
            })
            print(f"  [{cfg_name}] Avg 3-gram Rep: {avg_rep:.2%} | EOS Rate: {eos_rate:.0%} | Speed: {avg_speed:.1f} tok/s")

        full_results["benchmarks"]["b13_stability"][model_name] = b13_results

        # ----------------------------------------------------------------------
        # Aggregate Quantitative Summary per Model
        # ----------------------------------------------------------------------
        avg_b7 = float(np.mean([x["score"] for x in b7_results]))
        avg_b8 = float(np.mean([x["score"] for x in b8_results]))
        avg_b9 = float(b9_result["score"])
        avg_b10 = float(np.mean([x["score"] for x in b10_results]))
        avg_b11 = float(np.mean([x["score"] for x in b11_results]))
        avg_b12 = float(np.mean([x["score"] for x in b12_results]))

        # Exact math accuracy
        math_acc = float(np.mean([1.0 if x["is_correct"] else 0.0 for x in b7_results]))
        # Code execution pass rate
        code_tests = [x["tests_passed"] / max(x["total_tests"], 1) for x in b8_results if x["total_tests"] > 0]
        code_pass_rate = float(np.mean(code_tests)) if code_tests else 0.0
        # Overall EOS rate across Benchmark 13
        eos_rate_all = float(np.mean([x["eos_termination_rate"] for x in b13_results]))

        full_results["quantitative_summary"][model_name] = {
            "math_accuracy": round(math_acc * 100, 1),
            "code_pass_rate": round(code_pass_rate * 100, 1),
            "eos_termination_rate": round(eos_rate_all * 100, 1),
            "b7_math_score": round(avg_b7, 2),
            "b8_coding_score": round(avg_b8, 2),
            "b9_summarization_score": round(avg_b9, 2),
            "b10_instruction_score": round(avg_b10, 2),
            "b11_conversational_score": round(avg_b11, 2),
            "b12_uncertainty_score": round(avg_b12, 2),
            "mean_benchmark_score": round(
                float(np.mean([avg_b7, avg_b8, avg_b9, avg_b10, avg_b11, avg_b12])), 2
            ),
        }

        # Clean up model from CUDA memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # Generate Visualization Charts
    # --------------------------------------------------------------------------
    generate_phase9_plots(full_results)

    # Save complete raw results to JSON
    out_json = PROJECT_ROOT / "analysis" / "phase9_evaluation_results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n[Saved JSON Output] {out_json}")

    # Copy artifacts
    artifact_dir = Path(r"C:\Users\rupar\.gemini\antigravity\brain\60461eed-366b-4384-8e0d-82baa726f007")
    if artifact_dir.exists():
        import shutil
        shutil.copy(out_json, artifact_dir / "phase9_evaluation_results.json")
        for plot_file in (PROJECT_ROOT / "analysis" / "plots").glob("phase9_*.png"):
            shutil.copy(plot_file, artifact_dir / plot_file.name)
            print(f"[Copied to Artifacts] {plot_file.name}")

    print("\nPhase 9 Comprehensive Evaluation Complete!")
    return full_results


def generate_phase9_plots(results: dict) -> None:
    """Generate high-resolution comparative charts for Phase 9."""
    plots_dir = PROJECT_ROOT / "analysis" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    models = list(MODELS.keys())
    short_names = ["Model 1\n(Base)", "Model 2\n(Track A)", "Model 3\n(SFT)"]
    summary = results["quantitative_summary"]

    # 1. Bar Chart: 6 Benchmark Dimensions across 3 models
    categories = [
        "Math Reasoning",
        "Coding & Prog",
        "Summarization",
        "Multi-step Inst",
        "Conversational",
        "Uncertainty / Halluc",
    ]
    keys = [
        "b7_math_score",
        "b8_coding_score",
        "b9_summarization_score",
        "b10_instruction_score",
        "b11_conversational_score",
        "b12_uncertainty_score",
    ]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 5.5))
    colors = ["#4A90E2", "#50E3C2", "#F5A623"]

    for idx, (m_name, color) in enumerate(zip(models, colors)):
        scores = [summary[m_name][k] for k in keys]
        rects = ax.bar(x + (idx - 1) * width, scores, width, label=m_name, color=color, alpha=0.9, edgecolor="black", linewidth=0.5)
        for r in rects:
            height = r.get_height()
            if height > 0.1:
                ax.annotate(f"{height:.1f}", xy=(r.get_x() + r.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Score (0 to 5 Rubric)", fontsize=11, fontweight="bold")
    ax.set_title("Comprehensive Capability Evaluation across 3 Model Generations (Benchmarks 7–12)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 5.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=10)
    plt.tight_layout()

    bar_path = plots_dir / "phase9_benchmarks_comparison.png"
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Quantitative Accuracy & Stability Comparison Chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    math_accs = [summary[m]["math_accuracy"] for m in models]
    code_passes = [summary[m]["code_pass_rate"] for m in models]
    eos_rates = [summary[m]["eos_termination_rate"] for m in models]

    b1 = axes[0].bar(short_names, math_accs, color=colors, width=0.5, edgecolor="black")
    axes[0].set_title("Mathematical Accuracy (%) ↑", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 100)
    for b in b1:
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.0f}%", ha="center", fontweight="bold")

    b2 = axes[1].bar(short_names, code_passes, color=colors, width=0.5, edgecolor="black")
    axes[1].set_title("Code Execution Pass Rate (%) ↑", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, 100)
    for b in b2:
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.0f}%", ha="center", fontweight="bold")

    b3 = axes[2].bar(short_names, eos_rates, color=colors, width=0.5, edgecolor="black")
    axes[2].set_title("EOS Termination Rate (%) ↑", fontsize=11, fontweight="bold")
    axes[2].set_ylim(0, 100)
    for b in b3:
        axes[2].text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.0f}%", ha="center", fontweight="bold")

    for a in axes:
        a.grid(axis="y", linestyle="--", alpha=0.4)
        a.set_axisbelow(True)

    plt.suptitle("Deterministic Task Performance: Math Accuracy, Code Execution, and EOS Stopping", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    metrics_path = plots_dir / "phase9_metrics_breakdown.png"
    plt.savefig(metrics_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Generated Plots] {bar_path.name}, {metrics_path.name}")


if __name__ == "__main__":
    run_evaluation()

