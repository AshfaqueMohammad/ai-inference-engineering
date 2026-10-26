"""Run class1_homework.py on Modal (CPU, T4, L4, A10 GPUs).

Usage (from the repo root):
    modal run week1/class1_modal.py::check_gpu                 # quick environment check on a T4
    modal run week1/class1_modal.py                            # full lab on a T4 GPU
    modal run week1/class1_modal.py --hardware cpu             # full lab on CPU
    modal run week1/class1_modal.py --hardware all             # cpu + t4 + l4 + a10 in parallel
    modal run week1/class1_modal.py --hardware t4 --profile --kv-steps 5   # extra flags go to the script
"""

import argparse
import subprocess
import sys
from pathlib import Path

import modal

HERE = Path(__file__).parent
SCRIPT_REMOTE_PATH = "/root/class1_homework.py"
REPORT_REMOTE_PATH = "/tmp/class1_report.html"

app = modal.App("class1-prefill-decode")

# Container image: Python + pinned libraries + the homework script.
image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(
        "torch==2.14.1",
        "transformers==5.20.2",
        "pandas==3.1.0",
    )
    .env({"HF_HOME": "/cache/huggingface", "PYTHONUNBUFFERED": "1"})
    .add_local_file(HERE / "class1_homework.py", SCRIPT_REMOTE_PATH)
)

# Persistent volume so TinyLlama (~2.2 GB) and GPT-2 are downloaded only once.
hf_cache = modal.Volume.from_name("class1-hf-cache", create_if_missing=True)

COMMON = dict(image=image,
              volumes={"/cache": hf_cache},
              timeout=30 * 60,
              secrets=[modal.Secret.from_name("huggingface-secret")],
              )


def _run_script(runtime_label: str, script_args: list[str]) -> str:
    """Run the homework script inside the container and return the HTML report."""
    cmd = [
        sys.executable, SCRIPT_REMOTE_PATH,
        "--runtime", runtime_label,
        "--no-reflection",                 # no keyboard in a Modal container
        "--report", REPORT_REMOTE_PATH,
        *script_args,
    ]
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    hf_cache.commit()
    return Path(REPORT_REMOTE_PATH).read_text(encoding="utf-8")


@app.function(cpu=4, memory=16384, **COMMON)
def run_cpu(script_args: list[str]) -> str:
    return _run_script("Modal CPU (4 cores)", script_args)


@app.function(gpu="T4", **COMMON)
def run_t4(script_args: list[str]) -> str:
    return _run_script("Modal T4 GPU", script_args)


@app.function(gpu="L4", **COMMON)
def run_l4(script_args: list[str]) -> str:
    return _run_script("Modal L4 GPU", script_args)


@app.function(gpu="A10", **COMMON)
def run_a10(script_args: list[str]) -> str:
    return _run_script("Modal A10 GPU", script_args)


RUNNERS = {"cpu": run_cpu, "t4": run_t4, "l4": run_l4, "a10": run_a10}


@app.function(gpu="T4", image=image)
def check_gpu():
    """Print the GPU, driver, and library versions the lab will run with."""
    import torch
    import transformers

    subprocess.run(["nvidia-smi"], check=False)
    print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
    print("transformers", transformers.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0), "| compute capability:", torch.cuda.get_device_capability(0))
        print("compiled for:", torch.cuda.get_arch_list())


def _parse_args(arglist: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Split `--hardware` (ours) from everything else (forwarded to the script)."""
    parser = argparse.ArgumentParser(prog="modal run week1/class1_modal.py")
    parser.add_argument("--hardware", default="t4", choices=[*RUNNERS, "all"])
    args, script_args = parser.parse_known_args(list(arglist))
    targets = list(RUNNERS) if args.hardware == "all" else [args.hardware]
    return targets, script_args


@app.local_entrypoint()
def main(*arglist: str):
    targets, script_args = _parse_args(arglist)
    print(f"Hardware: {', '.join(targets)} | script args: {script_args or '(defaults)'}")

    # spawn() starts every target in parallel; get() waits for each result.
    calls = {hw: RUNNERS[hw].spawn(script_args) for hw in targets}

    out_dir = HERE / "reports"
    out_dir.mkdir(exist_ok=True)
    for hw, call in calls.items():
        try:
            html = call.get()
        except Exception as exc:
            print(f"[{hw}] FAILED: {exc}")
            continue
        path = out_dir / f"class1_report_{hw}.html"
        path.write_text(html, encoding="utf-8")
        print(f"[{hw}] report saved → {path}")
