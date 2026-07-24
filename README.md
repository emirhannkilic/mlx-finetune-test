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

## Status

- [x] mlx-lm installed, baseline inference tested
- [ ] LoRA fine-tuning on dummy data
- [ ] Post-fine-tune evaluation (baseline vs. fine-tuned outputs)
- [ ] Serving (not yet in scope)
