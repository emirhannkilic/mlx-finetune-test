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
scripts/    inference, evaluation, and pipeline scripts
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

## Evaluation

`scripts/evaluate.py` runs the fine-tuned adapter against prompts that are not in
the training set (cross-lingual, domain-shift, paraphrase — see Results below) and
checks that the `[SOC-TEST]` marker is still present in each completion:

```bash
python scripts/evaluate.py --adapter-path adapters
```

Prints a pass/fail summary and writes full completions to `outputs/eval_results.json`.

## Serving

MLX only runs on Apple Silicon, so it isn't a production serving target. To run the
fine-tuned model on a standard (e.g. Linux) server, the LoRA adapter is fused into the
base model and converted to GGUF for use with [llama.cpp](https://github.com/ggml-org/llama.cpp).

`scripts/run_pipeline.py` chains fuse → GGUF conversion → quantization → (optionally)
serving:

```bash
python scripts/run_pipeline.py --serve
```

It looks for a llama.cpp checkout via `--llama-cpp-path`, the `LLAMA_CPP_PATH`
env var, or `~/llama.cpp` (in that order). If none is found, it stops after the
MLX fuse step instead of failing — llama.cpp isn't pip-installable, so its
location varies by machine. Run `python scripts/run_pipeline.py --help` for all
options (`--adapter-path`, `--quant-type`, `--port`, ...).

<details>
<summary>Equivalent manual commands</summary>

**1. Fuse the adapter into the base model (dequantized, fp16):**
```bash
mlx_lm.fuse --model mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit \
  --adapter-path adapters --save-path fused_model --dequantize
```

**2. Convert to GGUF.** `mlx_lm.fuse --export-gguf` doesn't support the Qwen3
architecture, so conversion goes through llama.cpp's own script instead:
```bash
python ~/llama.cpp/convert_hf_to_gguf.py fused_model \
  --outfile fused_model/model.gguf --outtype f16
```

**3. Quantize** (fp16 → Q4_K_M, ~8GB → ~2.4GB):
```bash
~/llama.cpp/build/bin/llama-quantize fused_model/model.gguf \
  fused_model/model-q4_k_m.gguf Q4_K_M
```

**4. Serve** an OpenAI-compatible API:
```bash
~/llama.cpp/build/bin/llama-server -m fused_model/model-q4_k_m.gguf --port 8080
```
</details>

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Your question here"}], "max_tokens": 100}'
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
- **Post-serving**: after fusing, converting to GGUF, and quantizing to Q4_K_M, the
  marker was still present in responses served through llama.cpp's API — the full
  MLX-to-GGUF conversion chain preserves the fine-tuned behavior.


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
- [x] Serving: fuse → GGUF → quantize → llama.cpp server → API verification
- [ ] Real dataset selection
- [ ] Production serving config (auth, concurrency, deployment target)
