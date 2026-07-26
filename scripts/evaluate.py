"""Runs the fine-tuned adapter against hand-written phishing/safe emails that
are not verbatim in data/train.jsonl and checks whether the predicted class
(Phishing/Safe) matches the expected one — the generalization check that was
previously done by hand (see README "Results").
"""

import argparse
import json
from pathlib import Path

from mlx_lm import load, generate

MODEL = "mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit"
LABELS = ("Phishing", "Safe")

EVAL_PROMPTS = [
    {
        "category": "phishing-prize-scam",
        "prompt": "Congratulations! You have won a $1000 gift card. Click here immediately to claim your prize before it expires: http://bit.ly/claim-now",
        "expected": "Phishing",
    },
    {
        "category": "phishing-account-threat",
        "prompt": "URGENT: Your account will be suspended. Verify your password now at http://secure-bank-login.ru/verify",
        "expected": "Phishing",
    },
    {
        "category": "phishing-identity-theft",
        "prompt": "Dear customer, we detected unusual activity. Please confirm your identity by entering your SSN and card number at http://verify-now.info",
        "expected": "Phishing",
    },
    {
        "category": "safe-work",
        "prompt": "Hi team, attached is the quarterly report for review. Let me know if you have questions before Friday.",
        "expected": "Safe",
    },
    {
        "category": "safe-personal-reminder",
        "prompt": "Reminder: our dentist appointment is on Tuesday at 3pm.",
        "expected": "Safe",
    },
    {
        "category": "safe-casual",
        "prompt": "Lunch tomorrow at noon? Let me know what you feel like.",
        "expected": "Safe",
    },
]


def extract_label(completion):
    for label in LABELS:
        if label in completion:
            return label
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", default="adapters")
    parser.add_argument("--output", default="outputs/eval_results.json")
    args = parser.parse_args()

    model, tokenizer = load(MODEL, adapter_path=args.adapter_path)

    results = []
    for case in EVAL_PROMPTS:
        messages = [{"role": "user", "content": case["prompt"]}]
        formatted = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        completion = generate(model, tokenizer, prompt=formatted, verbose=False, max_tokens=15)
        predicted = extract_label(completion)
        correct = predicted == case["expected"]
        results.append(
            {
                "category": case["category"],
                "prompt": case["prompt"],
                "expected": case["expected"],
                "completion": completion,
                "predicted": predicted,
                "correct": correct,
            }
        )
        status = "PASS" if correct else "FAIL"
        print(f"[{status}] ({case['category']}) expected={case['expected']} predicted={predicted}")
        print(f"       -> {completion!r}\n")

    passed = sum(r["correct"] for r in results)
    print(f"{passed}/{len(results)} passed")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
