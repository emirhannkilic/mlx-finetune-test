"""Runs the fine-tuned adapter against out-of-distribution prompts and checks
that the [SOC-TEST] marker still shows up — the generalization check that was
previously done by hand (see README "Results").
"""

import argparse
import json
from pathlib import Path

from mlx_lm import load, generate

MODEL = "mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit"
MARKER = "[SOC-TEST]"

# None of these prompts appear verbatim in data/train.jsonl.
EVAL_PROMPTS = [
    # cross-lingual: training data is entirely in English
    {"category": "cross-lingual", "prompt": "Fransa'nın başkenti neresidir?"},
    # domain-shift: unrelated to any training topic, tests SOC-relevant knowledge
    {"category": "domain-shift", "prompt": "What is a common indicator of a phishing email?"},
    # paraphrase of a training example ("How many days are in a week?")
    {"category": "paraphrase", "prompt": "Tell me the number of days that make up one week."},
]


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
        completion = generate(model, tokenizer, prompt=formatted, verbose=False)
        marker_found = MARKER in completion
        results.append(
            {
                "category": case["category"],
                "prompt": case["prompt"],
                "completion": completion,
                "marker_found": marker_found,
            }
        )
        status = "PASS" if marker_found else "FAIL"
        print(f"[{status}] ({case['category']}) {case['prompt']!r}")
        print(f"       -> {completion!r}\n")

    passed = sum(r["marker_found"] for r in results)
    print(f"{passed}/{len(results)} passed")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
