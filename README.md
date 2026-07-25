# mlx-finetune-test

A hands-on pipeline for LoRA fine-tuning of a local LLM on Apple Silicon using
[MLX](https://github.com/ml-explore/mlx). This is a learning project focused on getting a
fine-tuning + inference pipeline working end-to-end, not on data quality or model performance.

## Why MLX

[Unsloth](https://github.com/unslothai/unsloth) was evaluated first but depends on
CUDA/Triton and does not run natively on Apple Silicon. MLX is Apple's own array framework
with first-class support on M-series chips, so it was chosen instead.

## Model

[`mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit`](https://huggingface.co/mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit)
— a 4-bit MLX conversion of an abliterated Qwen3-4B (2.26GB). Abliteration removes the
model's refusal behavior; it does not add new knowledge or capability.

Model weights are not stored in this repo — they are downloaded on first run and cached
under `~/.cache/huggingface/`.

## Project structure

```
scripts/    inference and fine-tuning scripts
data/       training/validation data (train.jsonl, valid.jsonl) — not tracked in git
adapters/   LoRA adapter output — not tracked in git
outputs/    logs and experiment output — not tracked in git
```

Data format for `data/train.jsonl` and `data/valid.jsonl`: one JSON object per line,
`{"prompt": "...", "completion": "..."}`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Fine-tuning

The dummy dataset trains the model to prepend a fixed `[SOC-TEST]` marker to every
response — a behavior the base model doesn't exhibit, chosen so a successful fine-tune
is visible in the output, not just in the loss curve.

```bash
mlx_lm.lora --model mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit \
  --train --data ./data --adapter-path adapters --iters 200 --batch-size 2
```

Test the fine-tuned adapter:

```bash
mlx_lm.generate --model mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit \
  --adapter-path adapters --prompt "Your question here"
```

## Results

Val loss dropped from 6.589 to 0.504 over 200 iterations (25 train / 5 valid examples).
The adapter was tested against prompts outside the training set:

- **Cross-lingual**: a Turkish prompt still produced the marker and a correct answer,
  despite training data being entirely in English.
- **Domain shift**: a phishing-related prompt (unrelated to training topics) produced
  the marker along with an accurate, coherent answer — the base model's knowledge held
  up, the adapter didn't overwrite it.
- **Paraphrase**: a reworded version of a training question was answered correctly,
  ruling out simple memorization.


## Notes

The base model emits a `<think>...</think>` reasoning block by default, even for trivial
prompts. After fine-tuning on a dataset with no reasoning traces in the completions, the
model stopped producing them entirely (empty `<think></think>`). Fine-tuning on a
narrow behavior pattern can suppress unrelated default behaviors — worth keeping in mind
for future runs where reasoning output matters.


## Status

- [x] mlx-lm installed, baseline inference tested
- [x] LoRA fine-tuning on dummy data
- [x] Post-fine-tune evaluation (baseline vs. fine-tuned outputs, cross-lingual and
      domain-shift generalization confirmed)
- [ ] Real dataset selection
- [ ] Serving (not yet in scope)
