"""Chains the serving pipeline that was previously run as separate manual commands:

    mlx_lm.fuse --dequantize  ->  llama.cpp/convert_hf_to_gguf.py  ->  llama-quantize  ->  llama-server

mlx_lm.fuse --export-gguf doesn't support the Qwen3 architecture ("Model type
qwen3 not supported for GGUF conversion"), so the fp16 GGUF conversion and
quantization go through llama.cpp instead (see CLAUDE.md "Kilitli kararlar").

llama.cpp is not pip-installable (git clone + cmake build) and isn't part of
this repo, so its location varies by machine. If it can't be found, this
script warns and stops after the MLX fuse step instead of failing with a bare
FileNotFoundError.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

MODEL = "mlx-community/Josiefied-Qwen3-4B-abliterated-v1-4bit"


def run(cmd, **kwargs):
    print(f"$ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def find_llama_cpp(cli_arg):
    candidate = Path(cli_arg or os.environ.get("LLAMA_CPP_PATH", "~/llama.cpp")).expanduser()
    if not candidate.is_dir():
        return None
    if not (candidate / "convert_hf_to_gguf.py").exists():
        return None
    return candidate


def find_llama_binary(llama_cpp_path, name):
    for candidate in (llama_cpp_path / "build" / "bin" / name, llama_cpp_path / name):
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter-path", default="adapters")
    parser.add_argument("--fused-path", default="fused_model")
    parser.add_argument("--quant-type", default="Q4_K_M")
    parser.add_argument("--llama-cpp-path", default=None, help="Overrides LLAMA_CPP_PATH env var and the ~/llama.cpp default.")
    parser.add_argument("--serve", action="store_true", help="Launch llama-server after quantizing.")
    parser.add_argument("--port", default="8080")
    parser.add_argument(
        "--cors-origins",
        default="localhost",
        help="Passed to llama-server --cors-origins. Default 'localhost' (vs. llama-server's own default of "
        "'*', which allows any site to call the API from a user's browser). Use '*' to opt back into that.",
    )
    args = parser.parse_args()

    fused_path = Path(args.fused_path)

    print("=== Step 1/4: mlx_lm.fuse (LoRA adapter -> fp16 safetensors) ===")
    run([
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", MODEL,
        "--adapter-path", args.adapter_path,
        "--save-path", str(fused_path),
        "--dequantize",
    ])

    llama_cpp_path = find_llama_cpp(args.llama_cpp_path)
    if llama_cpp_path is None:
        print(
            "\nllama.cpp not found (checked --llama-cpp-path, LLAMA_CPP_PATH, and ~/llama.cpp).\n"
            "Stopping after the MLX fuse step — see README for llama.cpp setup instructions.\n"
            f"fp16 model is available at: {fused_path}"
        )
        return

    fp16_gguf = fused_path / "model.gguf"
    print(f"\n=== Step 2/4: convert_hf_to_gguf.py (fp16 safetensors -> {fp16_gguf.name}) ===")
    run([
        sys.executable, str(llama_cpp_path / "convert_hf_to_gguf.py"),
        str(fused_path),
        "--outfile", str(fp16_gguf),
    ])

    llama_quantize = find_llama_binary(llama_cpp_path, "llama-quantize")
    if llama_quantize is None:
        print(
            f"\nllama-quantize binary not found under {llama_cpp_path} (expected build/bin/llama-quantize).\n"
            "Build llama.cpp with cmake first. Stopping after the fp16 GGUF conversion.\n"
            f"fp16 GGUF is available at: {fp16_gguf}"
        )
        return

    quant_gguf = fused_path / f"model-{args.quant_type.lower()}.gguf"
    print(f"\n=== Step 3/4: llama-quantize (fp16 -> {args.quant_type}) ===")
    run([str(llama_quantize), str(fp16_gguf), str(quant_gguf), args.quant_type])
    print(f"\nQuantized model ready at: {quant_gguf}")

    if not args.serve:
        return

    llama_server = find_llama_binary(llama_cpp_path, "llama-server")
    if llama_server is None:
        print(f"\nllama-server binary not found under {llama_cpp_path}. Skipping serve step.")
        return

    server_cmd = [
        str(llama_server), "-m", str(quant_gguf),
        "--port", args.port,
        "--cors-origins", args.cors_origins,
    ]
    api_key = os.environ.get("LLAMA_API_KEY")
    if api_key:
        server_cmd += ["--api-key", api_key]
        print(f"\n=== Step 4/4: llama-server (port {args.port}, API key set via LLAMA_API_KEY) ===")
    else:
        print(
            f"\n=== Step 4/4: llama-server (port {args.port}, no API key) ===\n"
            "LLAMA_API_KEY not set — server will accept unauthenticated requests. "
            "Set it (export LLAMA_API_KEY=...) before running with --serve to require a key."
        )
    run(server_cmd)


if __name__ == "__main__":
    main()
