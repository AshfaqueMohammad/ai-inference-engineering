#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class 1: Prefill vs Decode — LLM Inference on CPU, TPU, and GPU.

Standalone script version of class1.py (originally the Colab notebook class1.ipynb:
https://colab.research.google.com/drive/1jC248ygTSNtznoJ8sEHCLeTvFJkJPg_u).

**Objective:** Run the same LLM inference code on **CPU**, **TPU**, and **T4 GPU** and
compare **TinyLlama-1.1B** vs **GPT-2**. Manually separate and time the **prefill** and
**decode** phases to observe how each phase behaves differently on different hardware.

Setup (Colab usually has PyTorch pre-installed):
    pip install torch transformers accelerate pandas

Usage:
    python week1/class1_homework.py                         # full run, label = detected device
    python week1/class1_homework.py --runtime "T4 GPU"      # custom runtime label
    python week1/class1_homework.py --profile               # profile the long prompt
    python week1/class1_homework.py --no-reflection --skip-kv-compare
    python week1/class1_homework.py --help
"""

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional TPU support (only available when the runtime is TPU)
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    TPU_AVAILABLE = True
except ImportError:
    TPU_AVAILABLE = False

# Optional Colab download support
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


# =============================================================================
# Configuration
# =============================================================================
#
# We compare two models on the **same prompts** and **same hardware**:
#
# | Model               | Params | Notes                                     |
# |---------------------|--------|-------------------------------------------|
# | TinyLlama-1.1B-Chat | 1.1B   | Chat-tuned, uses chat template            |
# | GPT-2               | 124M   | Classic baseline, plain text tokenization |

MAX_NEW_TOKENS = 50

MODELS = {
    "TinyLlama-1.1B": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "GPT-2": "gpt2",
}

SHORT_PROMPT = (
    "Explain the difference between prefill and decode in LLM inference in one paragraph."
)
LONG_PROMPT = SHORT_PROMPT + " " + " ".join([SHORT_PROMPT] * 8)

OPENAI_PRICING = {
    "gpt-4o": {"input_per_1m": 2.50, "cached_input_per_1m": 1.25, "output_per_1m": 10.00},
    "gpt-4o-mini": {"input_per_1m": 0.15, "cached_input_per_1m": 0.075, "output_per_1m": 0.60},
}
DEFAULT_PRICING_MODEL = "gpt-4o"

RUN_PROFILER = False  # True = profiler on long prompt only

REQUESTS_PER_DAY = 10_000
AVG_INPUT_TOKENS = 500
AVG_OUTPUT_TOKENS = 200

KV_COMPARE_STEPS = 10  # no-cache is slow — keep ≤20 on CPU

DEFAULT_REPORT_PATH = "class1_report.html"


def show_df(df: pd.DataFrame) -> None:
    """Script stand-in for the notebook's display(): print the whole table as text."""
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.to_string(index=False))


# =============================================================================
# 1. Device detection
# =============================================================================

def detect_available_devices() -> List[str]:
    """Return device names we can try, in priority order for this session."""
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if TPU_AVAILABLE:
        devices.append("tpu")
    devices.append("cpu")  # always available
    return devices


def resolve_device(device: str) -> Tuple[torch.device, str]:
    """
    Map a logical device name to a torch device + human-readable label.
    Raises clear errors if the requested device is not available.
    """
    device = device.lower()

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but no GPU found. In Colab: Runtime → Change runtime type → T4 GPU."
            )
        name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), f"GPU ({name})"

    if device == "tpu":
        if not TPU_AVAILABLE:
            raise RuntimeError(
                "TPU requested but torch_xla is unavailable. In Colab: Runtime → Change runtime type → TPU."
            )
        return xm.xla_device(), "TPU"

    if device == "cpu":
        return torch.device("cpu"), "CPU"

    raise ValueError(f"Unknown device '{device}'. Use 'cpu', 'cuda', or 'tpu'.")


# =============================================================================
# 2. Timing helpers
# =============================================================================
#
# Wall-clock timing plus an explicit device sync works on CPU, GPU, and TPU.
#
# Why this matters for serving: TTFT is dominated by prefill latency. Decode
# throughput (tokens/sec) drives how fast the rest of the response streams.

class Timer:
    """Wall-clock timer — reliable on CPU, GPU, and TPU."""

    def __init__(self, device: torch.device):
        self._start = None

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> float:
        if self._start is None:
            return 0.0
        return time.perf_counter() - self._start


def sync_device(device: torch.device) -> None:
    """Ensure async work is finished before measuring."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xla":
        xm.mark_step()


# =============================================================================
# 3. Load model and tokenizer
# =============================================================================

def tokenize_prompt(tokenizer, prompt: str, device: torch.device) -> dict:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    else:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.shape[-1] == 0:
        raise ValueError("Tokenization produced empty input_ids")
    return {"input_ids": input_ids.to(device)}


def load_model_and_tokenizer(model_name: str, device: torch.device):
    print(f"  loading tokenizer + weights → {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device)
    model.eval()
    print("  model ready.")
    return model, tokenizer


# =============================================================================
# 4. Prefill + decode with KV cache
# =============================================================================

@dataclass
class InferenceResult:
    device_label: str
    prompt_label: str
    prompt_tokens: int
    generated_tokens: int
    prefill_sec: float
    decode_sec: float
    ttft_sec: float
    decode_tokens_per_sec: float
    first_decode_step_sec: float
    generated_text: str = ""
    error: Optional[str] = None
    model_name: str = ""
    model_repo: str = ""
    recorded_at: str = ""

    def is_valid(self) -> bool:
        return self.error is None and self.prompt_tokens > 0 and self.generated_tokens > 0

    def status_label(self) -> str:
        if self.error:
            return f"ERR: {self.error[:60]}"
        if not self.is_valid():
            return "INVALID"
        return "OK"


def _profiler_activities(device: torch.device) -> List[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    return activities


def _get_past_key_values(outputs):
    if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
        return outputs.past_key_values
    if hasattr(outputs, "cache") and outputs.cache is not None:
        return outputs.cache
    return None


def _validate_generation_outputs(outputs, step_name: str) -> None:
    if outputs is None or outputs.logits is None:
        raise RuntimeError(f"{step_name}: no logits returned")
    if _get_past_key_values(outputs) is None:
        raise RuntimeError(f"{step_name}: KV cache missing (use_cache=True)")


def _print_token(tokenizer, token_id: int, prefix: str = "") -> None:
    piece = tokenizer.decode([token_id], skip_special_tokens=True)
    display = repr(piece) if piece.strip() else f"id={token_id}"
    print(f"{prefix}token {display}")


def prefill_and_decode(
    model, tokenizer, prompt: str, device: torch.device,
    model_label: str = "",
    max_new_tokens: int = MAX_NEW_TOKENS, use_profiler: bool = False, verbose: bool = True,
) -> InferenceResult:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")

    prompt_label = "long" if len(prompt) > len(SHORT_PROMPT) + 20 else "short"
    device_label = ""
    tag = f"{model_label} | " if model_label else ""

    try:
        if verbose:
            print(f"\n── {tag}{prompt_label.upper()} prompt ──")

        if verbose:
            print("  PREFILL [1/6] tokenize prompt ...")
        inputs = tokenize_prompt(tokenizer, prompt, device)
        prompt_len = int(inputs["input_ids"].shape[-1])
        if verbose:
            print(f"           → {prompt_len} input tokens on {device}")

        generated_ids: List[int] = []
        first_decode_step_sec = 0.0

        if verbose:
            print("  PREFILL [2/6] forward pass — all prompt tokens at once (parallel matmuls) ...")
        prefill_timer = Timer(device)
        prefill_timer.start()
        with torch.no_grad():
            if use_profiler:
                with profile(activities=_profiler_activities(device), record_shapes=True) as prefill_prof:
                    outputs = model(**inputs, use_cache=True)
            else:
                prefill_prof = None
                outputs = model(**inputs, use_cache=True)
        if verbose:
            print("  PREFILL [3/6] sync device (wait for GPU/TPU kernels) ...")
        sync_device(device)
        prefill_sec = prefill_timer.stop()
        if verbose:
            print(f"           → forward finished in {prefill_sec:.3f}s")

        if verbose:
            print("  PREFILL [4/6] validate logits + KV cache ...")
        _validate_generation_outputs(outputs, "Prefill")
        if verbose:
            print(f"           → logits shape {tuple(outputs.logits.shape)}")

        if verbose:
            print("  PREFILL [5/6] read past_key_values from output cache ...")
        past_key_values = _get_past_key_values(outputs)
        if verbose:
            print("           → KV cache ready for decode")

        if verbose:
            print("  PREFILL [6/6] argmax on last logit → first generated token ...")
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        first_id = int(next_token.item())
        generated_ids.append(first_id)
        if verbose:
            _print_token(tokenizer, first_id, prefix="           → first ")

        if verbose:
            print(f"\n  DECODE — generate up to {max_new_tokens - 1} more tokens (1 forward pass each) ...")
        decode_timer = Timer(device)
        decode_timer.start()

        def _decode_loop():
            nonlocal outputs, past_key_values, next_token, first_decode_step_sec
            for step_idx in range(max_new_tokens - 1):
                step_num = step_idx + 1
                if verbose:
                    print(f"  DECODE step {step_num}/{max_new_tokens - 1} — forward 1 new token + reuse KV cache ...")

                step_timer = Timer(device)
                step_timer.start()
                with torch.no_grad():
                    outputs = model(
                        input_ids=next_token.unsqueeze(1),
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                if verbose:
                    print("           sync device ...")
                sync_device(device)
                step_sec = step_timer.stop()
                if step_idx == 0:
                    first_decode_step_sec = step_sec

                if verbose:
                    print(f"           validate logits + updated KV cache ({step_sec:.3f}s)")
                _validate_generation_outputs(outputs, f"Decode step {step_num}")

                past_key_values = _get_past_key_values(outputs)
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1)
                token_id = int(next_token.item())
                generated_ids.append(token_id)
                if verbose:
                    _print_token(tokenizer, token_id, prefix="           → sampled ")

        if use_profiler:
            with profile(activities=_profiler_activities(device), record_shapes=True) as decode_prof:
                _decode_loop()
        else:
            decode_prof = None
            _decode_loop()

        decode_sec = decode_timer.stop()
        decode_steps = max(len(generated_ids) - 1, 0)
        decode_tps = decode_steps / decode_sec if decode_sec > 0 else 0.0
        ttft_sec = prefill_sec + first_decode_step_sec

        if verbose:
            print(f"\n  DONE — prefill {prefill_sec:.3f}s | decode {decode_sec:.3f}s | {decode_tps:.1f} tok/s")
            print(f"         TTFT = prefill + 1st decode step = {ttft_sec:.3f}s")
            print(f"         generated {len(generated_ids)} tokens total")

        result = InferenceResult(
            device_label=device_label,
            prompt_label=prompt_label,
            prompt_tokens=prompt_len,
            generated_tokens=len(generated_ids),
            prefill_sec=prefill_sec,
            decode_sec=decode_sec,
            ttft_sec=ttft_sec,
            decode_tokens_per_sec=decode_tps,
            first_decode_step_sec=first_decode_step_sec,
            generated_text=tokenizer.decode(generated_ids, skip_special_tokens=True),
            model_name=model_label,
        )

        if use_profiler and prefill_prof is not None:
            print_profiler_summary(prefill_prof, device, phase="PREFILL")
            print_profiler_summary(decode_prof, device, phase="DECODE")
        return result

    except Exception as exc:
        if verbose:
            print(f"  ERROR: failed: {exc}")
        return InferenceResult(
            device_label=device_label,
            prompt_label=prompt_label,
            prompt_tokens=0,
            generated_tokens=0,
            prefill_sec=0.0,
            decode_sec=0.0,
            ttft_sec=0.0,
            decode_tokens_per_sec=0.0,
            first_decode_step_sec=0.0,
            error=str(exc),
            model_name=model_label,
        )


# =============================================================================
# 5. Profiler — focus on attention and matmul
# =============================================================================
#
# We profile prefill and decode separately and filter output to ops related to
# matrix multiply and attention.

ATTENTION_MATMUL_KEYWORDS = (
    "mm", "bmm", "addmm", "matmul", "linear", "attention", "sdpa", "softmax", "scaled_dot_product",
)


def _is_attention_or_matmul_op(op_name: str) -> bool:
    name = op_name.lower()
    return any(k in name for k in ATTENTION_MATMUL_KEYWORDS)


def print_profiler_summary(prof, device: torch.device, phase: str = "") -> None:
    if prof is None:
        return
    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    label = phase.upper() if phase else "RUN"
    print(f"\n{'='*60}\nProfiler — {label}\n{'='*60}")
    print(prof.key_averages().table(sort_by=sort_key, row_limit=15))
    filtered = [e for e in prof.key_averages() if _is_attention_or_matmul_op(e.key)]
    if filtered:
        print(f"\n--- {label}: attention/matmul ops ---")
        filtered.sort(key=lambda e: getattr(e, sort_key), reverse=True)
        for evt in filtered[:15]:
            print(f"  {evt.key[:55]:55s}  calls={evt.count}")


# =============================================================================
# 6. Run experiments
# =============================================================================

def print_result(r: InferenceResult, runtime_name: str = "") -> None:
    label = r.device_label or runtime_name or "?"
    model = r.model_name or "?"
    if r.error:
        print(f"\nERROR: [{label} | {model} | {r.prompt_label}] {r.error}")
        return
    if not r.is_valid():
        print(f"\nWARN: [{label} | {model} | {r.prompt_label}] invalid — tok={r.prompt_tokens}/{r.generated_tokens}")
        return
    print(f"\nSUMMARY [{label} | {model} | {r.prompt_label}]")
    print(f"   prefill {r.prefill_sec:.4f}s | decode {r.decode_sec:.4f}s | TTFT {r.ttft_sec:.4f}s | {r.decode_tokens_per_sec:.1f} tok/s")


def run_current_runtime(
    runtime_name: str,
    run_profiler: bool = RUN_PROFILER,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> List[InferenceResult]:
    print("\n" + "=" * 60)
    print(f"Starting run — runtime label: {runtime_name}")
    print("=" * 60)

    print("\nStep 1 — detect PyTorch backend for this session")
    device_name = detect_available_devices()[0]
    device, hw_label = resolve_device(device_name)
    print(f"  using {device_name} ({hw_label})")

    results: List[InferenceResult] = []
    prompts = [("short", SHORT_PROMPT, False), ("long", LONG_PROMPT, run_profiler)]

    for model_label, model_repo in MODELS.items():
        print("\n" + "=" * 60)
        print(f"MODEL: {model_label} ({model_repo})")
        print("=" * 60)

        print(f"\n  load {model_label} ...")
        model, tokenizer = load_model_and_tokenizer(model_repo, device)

        for label, prompt, use_prof in prompts:
            print(f"\n  run {model_label} | {label} prompt" + (" + profiler" if use_prof else ""))
            r = prefill_and_decode(
                model, tokenizer, prompt, device,
                model_label=model_label,
                max_new_tokens=max_new_tokens,
                use_profiler=use_prof,
                verbose=True,
            )
            r.device_label = runtime_name
            r.model_repo = model_repo
            r.recorded_at = datetime.now(timezone.utc).isoformat()
            print_result(r, runtime_name)
            results.append(r)

        print(f"\n  unload {model_label} + free memory")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


def print_results_summary(results: List[InferenceResult], runtime_name: str) -> None:
    print("\n" + "=" * 60)
    print(f"RESULTS — runtime: {runtime_name}")
    print("=" * 60)
    for r in results:
        if r.is_valid():
            print(
                f"  {r.model_name:16s} | {r.prompt_label:5s} | "
                f"prefill {r.prefill_sec:.4f}s | decode {r.decode_sec:.4f}s | "
                f"TTFT {r.ttft_sec:.4f}s | {r.decode_tokens_per_sec:.1f} tok/s"
            )
        else:
            print(f"  {r.model_name:16s} | {r.prompt_label:5s} | FAILED — {r.error or 'invalid'}")


# =============================================================================
# 7. Results table (current session)
# =============================================================================
#
# Shows results from this runtime only — 4 rows per run (2 models × 2 prompts).

def results_to_dataframe(results: List[InferenceResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Runtime": r.device_label,
            "Model": r.model_name,
            "Prompt": r.prompt_label,
            "Prompt Tok": r.prompt_tokens,
            "Prefill (s)": round(r.prefill_sec, 4) if r.is_valid() else None,
            "Decode (s)": round(r.decode_sec, 4) if r.is_valid() else None,
            "TTFT (s)": round(r.ttft_sec, 4) if r.is_valid() else None,
            "Tok/sec": round(r.decode_tokens_per_sec, 2) if r.is_valid() else None,
            "Status": r.status_label(),
        })
    return pd.DataFrame(rows)


# =============================================================================
# 8. Cost: OpenAI API pricing (input vs output tokens)
# =============================================================================
#
# Commercial APIs bill input and output tokens at different rates. We ran
# TinyLlama + GPT-2 locally (free compute), but apply OpenAI list prices as a
# proxy for what the same token counts would cost in production.

def estimate_request_cost(
    prompt_tokens: int,
    output_tokens: int,
    pricing_model: str = DEFAULT_PRICING_MODEL,
    verbose: bool = False,
) -> dict:
    """Map prefill tokens → input $, decode tokens → output $."""
    if pricing_model not in OPENAI_PRICING:
        raise ValueError(f"Unknown model '{pricing_model}'. Options: {list(OPENAI_PRICING)}")

    p = OPENAI_PRICING[pricing_model]
    input_cost = prompt_tokens * p["input_per_1m"] / 1_000_000
    output_cost = output_tokens * p["output_per_1m"] / 1_000_000
    total = input_cost + output_cost

    if verbose:
        print(f"    input  {prompt_tokens:4d} tok × ${p['input_per_1m']}/1M  = ${input_cost:.6f}  (prefill)")
        print(f"    output {output_tokens:4d} tok × ${p['output_per_1m']}/1M = ${output_cost:.6f}  (decode)")
        print(f"    total ${total:.6f}  ({output_cost/total*100:.0f}% from output)" if total else "")

    return {
        "pricing_model": pricing_model,
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total,
        "output_share_pct": (output_cost / total * 100) if total > 0 else 0.0,
        "price_ratio_output_to_input": p["output_per_1m"] / p["input_per_1m"],
    }


def results_to_cost_dataframe(
    results: List[InferenceResult],
    pricing_model: str = DEFAULT_PRICING_MODEL,
) -> pd.DataFrame:
    rows = []
    for r in results:
        if r.error or not r.is_valid():
            rows.append({
                "Device": r.device_label, "Model": r.model_name, "Prompt": r.prompt_label,
                "Input tok": r.prompt_tokens, "Output tok": r.generated_tokens,
                "Input $": None, "Output $": None, "Total $": None, "Output % of $": None,
            })
            continue
        c = estimate_request_cost(r.prompt_tokens, r.generated_tokens, pricing_model)
        rows.append({
            "Device": r.device_label, "Model": r.model_name, "Prompt": r.prompt_label,
            "Input tok": c["input_tokens"], "Output tok": c["output_tokens"],
            "Input $": round(c["input_cost_usd"], 6),
            "Output $": round(c["output_cost_usd"], 6),
            "Total $": round(c["total_cost_usd"], 6),
            "Output % of $": round(c["output_share_pct"], 1),
        })
    return pd.DataFrame(rows)


def run_cost_analysis(
    results: List[InferenceResult],
    pricing_model: str = DEFAULT_PRICING_MODEL,
    requests_per_day: int = REQUESTS_PER_DAY,
    avg_input_tokens: int = AVG_INPUT_TOKENS,
    avg_output_tokens: int = AVG_OUTPUT_TOKENS,
) -> None:
    print("\nCost analysis config:")
    print("  pricing proxy:", pricing_model)
    print("  scale scenario:", f"{requests_per_day:,} req/day, {avg_input_tokens} in + {avg_output_tokens} out tok")

    print("\nStep 1/3 — OpenAI list prices")
    print("(prefill tokens bill as INPUT, generated tokens bill as OUTPUT)\n")

    rows = []
    for name, p in OPENAI_PRICING.items():
        ratio = p["output_per_1m"] / p["input_per_1m"]
        rows.append({
            "Model": name,
            "Input ($/1M)": p["input_per_1m"],
            "Output ($/1M)": p["output_per_1m"],
            "Output/Input": f"{ratio:.0f}×",
        })
    show_df(pd.DataFrame(rows))
    print("\n→ Output tokens cost more — decode dominates $ at scale.")

    ok = [r for r in results if r.is_valid()]
    if not ok:
        print("\nWARN: No valid results — skipping per-run cost.")
    else:
        print(f"\nStep 2/3 — Cost per lab run (proxy: {pricing_model})")
        ratio = OPENAI_PRICING[pricing_model]["output_per_1m"] / OPENAI_PRICING[pricing_model]["input_per_1m"]
        print(f"  output/input price ratio: {ratio:.0f}×\n")

        for r in ok:
            print(f"  {r.model_name} | {r.prompt_label} prompt:")
            estimate_request_cost(r.prompt_tokens, r.generated_tokens, pricing_model, verbose=True)
            print()

        print("Summary table:")
        show_df(results_to_cost_dataframe(ok, pricing_model))

        example = max(ok, key=lambda r: r.generated_tokens)
        c = estimate_request_cost(example.prompt_tokens, example.generated_tokens, pricing_model)
        print(
            f"\nExample: {example.model_name} {example.prompt_label} → "
            f"${c['total_cost_usd']:.6f} total ({c['output_share_pct']:.0f}% from decode/output tokens)"
        )

    print(f"\nStep 3/3 — Scale calculator ({pricing_model})")
    print(f"  {requests_per_day:,} requests/day × {avg_input_tokens} input + {avg_output_tokens} output tokens\n")

    per = estimate_request_cost(avg_input_tokens, avg_output_tokens, pricing_model, verbose=True)
    daily_in = requests_per_day * avg_input_tokens * OPENAI_PRICING[pricing_model]["input_per_1m"] / 1_000_000
    daily_out = requests_per_day * avg_output_tokens * OPENAI_PRICING[pricing_model]["output_per_1m"] / 1_000_000
    print(f"\nDaily @ {requests_per_day:,} req: ${daily_in + daily_out:,.2f}  (input ${daily_in:,.2f} + output ${daily_out:,.2f})")
    print(f"Monthly (30d): ${(daily_in + daily_out) * 30:,.2f}")
    print(f"\n→ Output is {per['output_share_pct']:.0f}% of cost but only {avg_output_tokens/(avg_input_tokens+avg_output_tokens)*100:.0f}% of tokens.")


# =============================================================================
# 9. Your reflection
# =============================================================================

def _ask(question: str) -> str:
    try:
        return input(question).strip()
    except EOFError:
        return ""


def collect_reflection(runtime_name: str, interactive: bool = True) -> str:
    print("\n── Reflection ──\n")

    if interactive and sys.stdin.isatty():
        reflection_runtime = _ask("Runtime you tested (e.g. T4 GPU, CPU, TPU v5e-1): ")
        reflection_models = _ask("TinyLlama vs GPT-2 — what differed in prefill/decode timing? ")
        reflection_surprise = _ask("What surprised you about prefill vs decode? ")
    else:
        print("  (non-interactive — skipping prompts; reflection left blank)")
        reflection_runtime = reflection_models = reflection_surprise = ""

    reflection = (
        f"Runtime I tested: {reflection_runtime or runtime_name}\n\n"
        f"TinyLlama vs GPT-2:\n{reflection_models}\n\n"
        f"What surprised me about prefill vs decode?\n{reflection_surprise}"
    )

    print("\nSaved reflection preview:")
    print("-" * 40)
    print(reflection.strip())
    print("-" * 40)
    return reflection


# =============================================================================
# 10. Export report (HTML)
# =============================================================================
#
# Writes timings, costs, and the reflection to an HTML file.
# Open the HTML in a browser → Print → Save as PDF.

def build_html_report(
    results: List[InferenceResult],
    reflection: str,
    output_path: str = DEFAULT_REPORT_PATH,
    pricing_model: str = DEFAULT_PRICING_MODEL,
) -> Path:
    print("Building HTML report ...")
    path = Path(output_path)
    detail_df = results_to_dataframe(results)
    cost_df = results_to_cost_dataframe(results, pricing_model)
    print(f"  {len(detail_df)} timing rows, {len(cost_df)} cost rows")

    ok = [r for r in results if r.is_valid()]
    pivot_html = ""
    cost_pivot_html = ""
    if ok:
        df = pd.DataFrame([
            {"model": r.model_name, "prompt": r.prompt_label,
             "prefill_s": r.prefill_sec, "decode_s": r.decode_sec,
             "ttft_s": r.ttft_sec, "tok_per_sec": r.decode_tokens_per_sec}
            for r in ok
        ])
        for title, col in [
            ("Prefill (s)", "prefill_s"), ("Decode (s)", "decode_s"),
            ("TTFT (s)", "ttft_s"), ("Decode tok/sec", "tok_per_sec"),
        ]:
            pivot = df.pivot_table(index="prompt", columns="model", values=col, aggfunc="first")
            pivot = pivot.reindex(["short", "long"])
            pivot_html += f"<h3>{title}</h3>\n{pivot.round(4).to_html()}\n"

        cdf = pd.DataFrame([
            {"model": r.model_name, "prompt": r.prompt_label,
             **{k: v for k, v in estimate_request_cost(r.prompt_tokens, r.generated_tokens, pricing_model).items() if k.endswith("_usd")}}
            for r in ok
        ])
        for title, col in [
            ("Input cost — prefill ($)", "input_cost_usd"),
            ("Output cost — decode ($)", "output_cost_usd"),
            ("Total cost ($)", "total_cost_usd"),
        ]:
            pivot = cdf.pivot_table(index="prompt", columns="model", values=col, aggfunc="first")
            pivot = pivot.reindex(["short", "long"])
            cost_pivot_html += f"<h3>{title}</h3>\n{pivot.apply(lambda c: c.map(lambda x: f'${x:.6f}')).to_html()}\n"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Class 1 Report — Prefill vs Decode</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; }}
  h1 {{ border-bottom: 2px solid #333; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
  th {{ background: #f0f0f0; }}
  td:first-child, th:first-child {{ text-align: left; }}
  .reflection {{ background: #fafafa; border-left: 4px solid #0066cc; padding: 1rem; }}
</style></head><body>
<h1>Class 1: Prefill vs Decode</h1>
<p><strong>Lab models:</strong> {", ".join(f"{k} ({v})" for k, v in MODELS.items())}<br>
<strong>Cost proxy:</strong> {pricing_model}<br>
<strong>Generated:</strong> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
<h2>Timing — side by side</h2>
{pivot_html or '<p>No successful results yet.</p>'}
<h2>Cost — side by side</h2>
{cost_pivot_html or '<p>No cost data yet.</p>'}
<h2>All runs (timing)</h2>
{detail_df.to_html(index=False)}
<h2>All runs (cost)</h2>
{cost_df.to_html(index=False)}
<h2>Reflection</h2>
<div class="reflection"><pre style="white-space: pre-wrap; margin:0;">{reflection.strip()}</pre></div>
</body></html>"""

    path.write_text(html, encoding="utf-8")
    print(f"  written → {path.resolve()}")
    return path


def download_html_report(
    results: List[InferenceResult],
    reflection: str,
    output_path: str = DEFAULT_REPORT_PATH,
    pricing_model: str = DEFAULT_PRICING_MODEL,
) -> Optional[Path]:
    if not any(r.is_valid() for r in results):
        print("WARN: Cannot export — no valid results.")
        return None
    path = build_html_report(results, reflection=reflection, output_path=output_path, pricing_model=pricing_model)
    if IN_COLAB:
        try:
            files.download(str(path))
            print("Downloaded — open in browser, Print → Save as PDF")
            return path
        except Exception as exc:  # files.download only works from a notebook kernel
            print(f"  (Colab download unavailable from a script: {exc})")
    print("Open the HTML in a browser → Print → Save as PDF")
    return path


# =============================================================================
# 11. With vs without KV cache + memory
# =============================================================================

@dataclass
class KVCacheCompareResult:
    runtime: str
    model_name: str
    decode_steps: int
    with_cache_sec: float
    without_cache_sec: float
    prompt_tokens: int = 0
    cached_tokens: int = 0
    kv_layout: str = ""
    num_layers: int = 0
    num_kv_heads: int = 0
    kv_cache_bytes: int = 0
    kv_bytes_per_token: float = 0.0
    error: Optional[str] = None

    def is_valid(self) -> bool:
        return self.error is None and self.with_cache_sec > 0 and self.without_cache_sec > 0

    @property
    def speedup(self) -> float:
        return self.without_cache_sec / self.with_cache_sec if self.with_cache_sec > 0 else 0.0

    @property
    def kv_cache_mb(self) -> float:
        return self.kv_cache_bytes / (1024 * 1024)


def kv_layout_info(model) -> dict:
    cfg = model.config
    n_q = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_q)
    layout = f"GQA ({n_kv} KV / {n_q} Q)" if n_kv < n_q else f"MHA ({n_kv} KV heads)"
    return {"layout": layout, "num_layers": cfg.num_hidden_layers, "num_kv_heads": n_kv,
            "head_dim": cfg.hidden_size // n_q}


def kv_cache_nbytes(past_key_values) -> int:
    if past_key_values is None:
        return 0
    total = 0
    if hasattr(past_key_values, "key_cache"):
        for t in past_key_values.key_cache + past_key_values.value_cache:
            if t is not None:
                total += t.numel() * t.element_size()
        return total
    for layer in past_key_values:
        if isinstance(layer, (tuple, list)):
            for t in layer:
                if torch.is_tensor(t):
                    total += t.numel() * t.element_size()
    return total


def kv_cache_seq_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if hasattr(past_key_values, "key_cache") and past_key_values.key_cache:
        return int(past_key_values.key_cache[0].shape[-2])
    if past_key_values and isinstance(past_key_values[0], (tuple, list)):
        return int(past_key_values[0][0].shape[-2])
    return 0


def compare_kv_decode(model, tokenizer, prompt, device, model_label="", steps=KV_COMPARE_STEPS, verbose=True):
    layout = kv_layout_info(model)
    try:
        if verbose:
            print(f"\n  {model_label} | {layout['layout']} | {layout['num_layers']} layers")

        inputs = tokenize_prompt(tokenizer, prompt, device)
        input_ids = inputs["input_ids"]

        if verbose:
            print("  [with cache] prefill ...")
        with torch.no_grad():
            pref = model(**inputs, use_cache=True)
        past = _get_past_key_values(pref)
        next_tok = torch.argmax(pref.logits[:, -1, :], dim=-1)

        if verbose:
            print(f"  [with cache] {steps} decode steps (1 token each) ...")
        t_with = Timer(device); t_with.start()
        for step in range(1, steps + 1):
            if verbose:
                print(f"    step {step}/{steps} — 1 token + past_key_values")
            with torch.no_grad():
                out = model(input_ids=next_tok.unsqueeze(1), past_key_values=past, use_cache=True)
            sync_device(device)
            past = _get_past_key_values(out)
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1)
        with_sec = t_with.stop()

        cached = kv_cache_seq_len(past)
        kv_bytes = kv_cache_nbytes(past)
        bpt = kv_bytes / cached if cached else 0
        if verbose:
            print(f"    → {with_sec:.3f}s | KV cache {kv_bytes/1024/1024:.3f} MB ({bpt:.0f} B/token)")

        if verbose:
            print(f"  [no cache] {steps} decode steps (full sequence each) ...")
        seq = input_ids.clone()
        with torch.no_grad():
            out = model(input_ids=seq, use_cache=False)
        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1)
        t_without = Timer(device); t_without.start()
        for step in range(1, steps + 1):
            if verbose:
                print(f"    step {step}/{steps} — forward {seq.shape[-1]+1} tokens")
            seq = torch.cat([seq, next_tok.unsqueeze(1)], dim=1)
            with torch.no_grad():
                out = model(input_ids=seq, use_cache=False)
            sync_device(device)
            next_tok = torch.argmax(out.logits[:, -1, :], dim=-1)
        without_sec = t_without.stop()
        if verbose:
            print(f"    → {without_sec:.3f}s | speedup {without_sec/with_sec:.1f}× with cache")

        return KVCacheCompareResult(
            runtime="", model_name=model_label, decode_steps=steps,
            with_cache_sec=with_sec, without_cache_sec=without_sec,
            prompt_tokens=int(input_ids.shape[-1]), cached_tokens=cached,
            kv_layout=layout["layout"], num_layers=layout["num_layers"],
            num_kv_heads=layout["num_kv_heads"], kv_cache_bytes=kv_bytes, kv_bytes_per_token=bpt,
        )
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return KVCacheCompareResult("", model_label, steps, 0, 0, error=str(exc), kv_layout=layout.get("layout", ""))


def kv_cache_to_dataframe(results: List[KVCacheCompareResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Model": r.model_name, "KV layout": r.kv_layout, "Layers": r.num_layers,
        "KV heads": r.num_kv_heads, "Cached tok": r.cached_tokens,
        "KV cache (MB)": round(r.kv_cache_mb, 3) if r.is_valid() else None,
        "Bytes/token": round(r.kv_bytes_per_token, 0) if r.is_valid() else None,
        "With cache (s)": round(r.with_cache_sec, 4) if r.is_valid() else None,
        "Without cache (s)": round(r.without_cache_sec, 4) if r.is_valid() else None,
        "Speedup (×)": round(r.speedup, 2) if r.is_valid() else None,
    } for r in results])


def run_kv_comparison(runtime_label: str, steps: int = KV_COMPARE_STEPS) -> List[KVCacheCompareResult]:
    print(f"\nKV cache comparison — {steps} decode steps on short prompt\n")
    device_name = detect_available_devices()[0]
    device, _ = resolve_device(device_name)

    kv_results: List[KVCacheCompareResult] = []
    for model_label, model_repo in MODELS.items():
        print("\n" + "=" * 60)
        print(f"MODEL: {model_label}")
        print("=" * 60)
        model, tokenizer = load_model_and_tokenizer(model_repo, device)
        kv = compare_kv_decode(model, tokenizer, SHORT_PROMPT, device, model_label=model_label, steps=steps, verbose=True)
        kv.runtime = runtime_label
        kv_results.append(kv)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    df = kv_cache_to_dataframe(kv_results)
    print()
    show_df(df)

    if len(df) >= 2 and df["KV cache (MB)"].notna().all():
        gpt2 = df[df["Model"] == "GPT-2"].iloc[0]
        tiny = df[df["Model"] == "TinyLlama-1.1B"].iloc[0]
        print(
            f"\nMemory takeaway: GPT-2 ({gpt2['KV layout']}) uses {gpt2['Bytes/token']:.0f} B/token cached; "
            f"TinyLlama ({tiny['KV layout']}) uses {tiny['Bytes/token']:.0f} B/token. "
            f"Grouped KV heads → smaller cache even though TinyLlama has more layers."
        )
    return kv_results


# =============================================================================
# Main
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Class 1: Prefill vs Decode — time LLM inference phases on CPU, TPU, or GPU.",
    )
    parser.add_argument("--runtime", default=None,
                        help="Label for this runtime (e.g. 'T4 GPU', 'CPU', 'TPU v5e-1'). Defaults to the detected device.")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                        help=f"Tokens to generate per run (default: {MAX_NEW_TOKENS}).")
    parser.add_argument("--profile", action="store_true", default=RUN_PROFILER,
                        help="Run torch.profiler on the long prompt.")
    parser.add_argument("--pricing-model", choices=list(OPENAI_PRICING), default=DEFAULT_PRICING_MODEL,
                        help=f"OpenAI price list used as the cost proxy (default: {DEFAULT_PRICING_MODEL}).")
    parser.add_argument("--requests-per-day", type=int, default=REQUESTS_PER_DAY,
                        help=f"Scale scenario: requests per day (default: {REQUESTS_PER_DAY:,}).")
    parser.add_argument("--avg-input-tokens", type=int, default=AVG_INPUT_TOKENS,
                        help=f"Scale scenario: input tokens per request (default: {AVG_INPUT_TOKENS}).")
    parser.add_argument("--avg-output-tokens", type=int, default=AVG_OUTPUT_TOKENS,
                        help=f"Scale scenario: output tokens per request (default: {AVG_OUTPUT_TOKENS}).")
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH,
                        help=f"Output path for the HTML report (default: {DEFAULT_REPORT_PATH}).")
    parser.add_argument("--no-reflection", action="store_true",
                        help="Skip the interactive reflection prompts.")
    parser.add_argument("--skip-kv-compare", action="store_true",
                        help="Skip the with/without KV cache comparison.")
    parser.add_argument("--kv-steps", type=int, default=KV_COMPARE_STEPS,
                        help=f"Decode steps for the KV cache comparison (default: {KV_COMPARE_STEPS}).")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    print("PyTorch backend ready.")
    print("  cuda available:", torch.cuda.is_available())
    print("  TPU available:", TPU_AVAILABLE)

    _, hw_label = resolve_device(detect_available_devices()[0])
    runtime_name = (args.runtime or hw_label).strip()

    print("\nModels to compare:")
    for label, repo in MODELS.items():
        print(f"  {label:16s} → {repo}")
    print("  short prompt chars:", len(SHORT_PROMPT))
    print("  long prompt chars:", len(LONG_PROMPT))

    print("\nExperiment config:")
    print("  runtime label =", runtime_name)
    print("  RUN_PROFILER =", args.profile)
    print("  MAX_NEW_TOKENS =", args.max_new_tokens)
    print("  models:", list(MODELS.keys()))

    # 6. Run experiments
    results = run_current_runtime(runtime_name, run_profiler=args.profile, max_new_tokens=args.max_new_tokens)
    print_results_summary(results, runtime_name)

    # 7. Results table
    print(f"\nRuntime (your label): {runtime_name}\n")
    print("Timing results (TinyLlama vs GPT-2):")
    show_df(results_to_dataframe(results))

    # 8. Cost analysis
    run_cost_analysis(
        results,
        pricing_model=args.pricing_model,
        requests_per_day=args.requests_per_day,
        avg_input_tokens=args.avg_input_tokens,
        avg_output_tokens=args.avg_output_tokens,
    )

    # 9. Reflection
    reflection = collect_reflection(runtime_name, interactive=not args.no_reflection)

    # 10. Export report
    print("\nExporting report with your reflection ...")
    download_html_report(results, reflection, output_path=args.report, pricing_model=args.pricing_model)

    # 11. KV cache comparison
    if not args.skip_kv_compare:
        run_kv_comparison(runtime_name, steps=args.kv_steps)

    return 0 if any(r.is_valid() for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
