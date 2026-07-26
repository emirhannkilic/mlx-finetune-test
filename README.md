# mlx-finetune-test

A hands-on pipeline for LoRA fine-tuning of a local LLM on Apple Silicon using
[MLX](https://github.com/ml-explore/mlx). This is a learning project focused on getting a
fine-tuning + inference pipeline working end-to-end, not on data quality or model performance.

The project went through two phases: first a dummy dataset to validate that the
fine-tuning → evaluation → serving pipeline works end-to-end (see "Phase 1"
below), then a real dataset (phishing email classification) to test the same
pipeline on an actual task (see "Phase 2").

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
scripts/    inference, data prep, evaluation, and pipeline scripts
data/       training/validation data (train.jsonl, valid.jsonl) — not tracked in git
adapters/   LoRA adapter output — not tracked in git
outputs/    logs, experiment output, and checkpoint backups — not tracked in git
```

Data format for `data/train.jsonl` and `data/valid.jsonl`: one JSON object per line,
`{"prompt": "...", "completion": "..."}`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Phase 1: dummy data — validating the pipeline

Before using real data, a dummy dataset (25 train / 5 valid examples) was used to
confirm the fine-tuning → evaluation → serving chain works at all. The dataset
trained the model to prepend a fixed `[SOC-TEST]` marker to every response — a
behavior the base model doesn't exhibit, chosen so a successful fine-tune is
visible in the output, not just in the loss curve.

```bash
mlx_lm.lora --model mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit \
  --train --data ./data --adapter-path adapters --iters 200 --batch-size 2
```

Results: val loss dropped from 6.589 to 0.504 over 200 iterations. The adapter
was tested against prompts outside the training set:

- **Cross-lingual**: a Turkish prompt still produced the marker and a correct
  answer, despite training data being entirely in English.
- **Domain shift**: a phishing-related prompt (unrelated to training topics)
  produced the marker along with an accurate, coherent answer — the base
  model's knowledge held up, the adapter didn't overwrite it.
- **Paraphrase**: a reworded version of a training question was answered
  correctly, ruling out simple memorization.
- **Post-serving**: after fusing, converting to GGUF, and quantizing to
  Q4_K_M, the marker was still present in responses served through
  llama.cpp's API — the full MLX-to-GGUF conversion chain preserves the
  fine-tuned behavior.

This phase also surfaced a side effect worth noting: the base model emits a
`<think>...</think>` reasoning block by default, even for trivial prompts.
After fine-tuning on a dataset with no reasoning traces in the completions,
the model stopped producing them entirely (empty `<think></think>`).
Fine-tuning on a narrow behavior pattern can suppress unrelated default
behaviors.

## Phase 2: real data — phishing email classification

With the pipeline validated, the dummy task was replaced with a real one:
classifying emails as `Phishing` or `Safe`, using the Kaggle
["Phishing Email Detection"](https://www.kaggle.com/datasets/subhajournal/phishingemails)
dataset (18,650 emails, `Email Text` + `Email Type` columns).

### Data preparation

`scripts/prepare_data.py` converts the raw CSV into the `{"prompt": ..., "completion": ...}`
JSONL format, with cleaning steps driven by problems found in the raw data:

- **Empty rows** (19) and **duplicate email text** (1,068) are dropped —
  duplicates matter because leaving them in risks the same email landing in
  both train and valid, i.e. data leakage.
- Rows longer than `--max-length` (default 1000 chars) are dropped. The raw
  data's length tail runs up to 17M characters against a median of 880; an
  earlier attempt with a 5000-char cutoff made fine-tuning impractically slow.
- The train/valid split (90/10) is **stratified** by label so both sets keep
  the same Safe/Phishing ratio as the source data (~61/39).

```bash
python scripts/prepare_data.py --max-length 1000
```

### Fine-tuning

```bash
mlx_lm.lora --model mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit \
  --train --data ./data --adapter-path adapters --iters 1000 --batch-size 4
```

### A lesson in trusting val loss too much

Val loss did not decrease monotonically: 4.761 (start) → 3.033 (iter 200) →
3.223 (iter 400) → 3.190 (iter 600) → 2.955 (iter 700) → 3.151 (iter 1000).
The naive read — "iter 200 has the lowest loss, use that" — turned out to be
wrong: manually testing the iter 200 checkpoint showed it predicted `Safe`
for every email, including obvious phishing. It had learned the majority
class (Safe is ~61% of the data), not the actual distinction — a loss curve
can look good while the model has learned nothing useful.

Checkpoints from later in training (iter 700 and iter 1000) were tested
against 6 hand-written emails (3 phishing, 3 safe, none copied from the
training set) and both scored 6/6, correctly classifying content never seen
during training. **Lesson: don't select a checkpoint on val loss alone —
verify with actual generation output before deciding.** Iter 1000 was kept
as the final adapter.

## Evaluation

`scripts/evaluate.py` runs the fine-tuned adapter against the 6 hand-written
phishing/safe emails described above and checks the predicted label against
the expected one:

```bash
python scripts/evaluate.py --adapter-path adapters
```

Prints a pass/fail summary (6/6 on the current adapter) and writes full
completions to `outputs/eval_results.json`.

## Serving

MLX only runs on Apple Silicon, so it isn't a production serving target. To run the
fine-tuned model on a standard (e.g. Linux) server, the LoRA adapter is fused into the
base model and converted to GGUF for use with [llama.cpp](https://github.com/ggml-org/llama.cpp).
This chain has been verified end-to-end with both the Phase 1 (dummy) and Phase 2
(phishing) adapters.

`scripts/run_pipeline.py` chains fuse → GGUF conversion → quantization → (optionally)
serving:

```bash
export LLAMA_API_KEY="choose-a-key"  # optional but recommended, see below
python scripts/run_pipeline.py --serve
```

It looks for a llama.cpp checkout via `--llama-cpp-path`, the `LLAMA_CPP_PATH`
env var, or `~/llama.cpp` (in that order). If none is found, it stops after the
MLX fuse step instead of failing — llama.cpp isn't pip-installable, so its
location varies by machine. Run `python scripts/run_pipeline.py --help` for all
options (`--adapter-path`, `--quant-type`, `--port`, ...).

**Auth and CORS**: `llama-server` defaults to no API key and `--cors-origins '*'`
(any website can call it from a browser). `run_pipeline.py` narrows the CORS
default to `localhost`, and passes `--api-key` to `llama-server` if the
`LLAMA_API_KEY` env var is set — leave it unset to run without auth (e.g. for
local testing), or set it to require a bearer token on every request. Pass
`--cors-origins '*'` explicitly to opt back into the permissive default.

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
  -H "Authorization: Bearer $LLAMA_API_KEY" \
  -d '{"messages": [{"role": "user", "content": "Your question here"}], "max_tokens": 100}'
```

## Status

- [x] mlx-lm installed, baseline inference tested
- [x] Phase 1: LoRA fine-tuning on dummy `[SOC-TEST]` data, full pipeline verified
      (fine-tune → evaluate → fuse → GGUF → quantize → llama.cpp server → API)
- [x] Phase 2: real dataset (Kaggle phishing emails) selected, cleaned, and used
      for LoRA fine-tuning; checkpoint selected by verifying actual output, not
      just val loss
- [x] Serving chain (fuse → GGUF → quantize → llama.cpp) re-run and verified
      with the Phase 2 adapter — API correctly classified phishing/safe test emails
- [x] Serving config: optional API key auth (`LLAMA_API_KEY`), CORS restricted
      to `localhost` by default
