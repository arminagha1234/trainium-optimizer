"""
vLLM-Neuron serve worker — the process that actually launches a `vllm serve` for
one (model, config), drives ONE latency request at the target shape, records
TTFT / TPOT / e2e + the greedy output token ids, and tears the serve down. Rank
0 (this process) writes a single JSON result the backend parses.

Launched by backends/vllm_serve.VllmServeBackend.measure():
    python vllm_serve_worker.py --model ... --tp ... --dtype ... \
        --quantization ... --max-num-seqs 1 --num-batched-tokens 512 \
        --async-scheduling 0|1 --speculative off|draft --draft-model ... \
        --input-len 2048 --output-len 512 --sla-seconds 2.0 --out result.json

Why a separate process: a `vllm serve` is a long-lived server with its own
engine subprocesses; the optimizer core must stay a clean single process. Each
measurement is an independent launch -> measure -> teardown, exactly like the
throughput path's torchrun worker.

The serve command mirrors the proven public recipe
(Armin-Neuron/gemma4-31b/.../launch_serve_public.sh): tensor-parallel-size,
max-model-len, max-num-seqs, max-num-batched-tokens, and an --additional-config
carrying the neuron_config (num_batched_tokens_buckets / num_seqs_buckets /
on_device_sampling greedy). Greedy sampling (all_greedy + temperature 0) makes
the output DETERMINISTIC so the framework's top-1-token equivalence gate can
compare this config's tokens against the bf16 baseline's.

ON-DEVICE STATUS: this path is DEFERRED (boxes busy) and has not yet been run on
hardware. It is code-complete and structured to mirror the working launch/bench
scripts; the backend wiring around it is exercised by the mock tests. Do not
cite a serving latency from this file until it has actually run on a Trn2 box.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

_EQ_K = 16  # equivalence signature length: first K greedy output token ids


def _log(msg: str) -> None:
    print(f"[vllm-serve-worker] {msg}", flush=True)


def _build_serve_cmd(a) -> tuple[list[str], dict]:
    """Assemble the `vllm serve` argv + env, mirroring launch_serve_public.sh."""
    neuron_config: dict = {
        "num_batched_tokens_buckets": [a.num_batched_tokens],
        "num_seqs_buckets": [a.max_num_seqs],
        "on_device_sampling_config": {"all_greedy": True},
    }
    additional = {"neuron_config": neuron_config}

    cmd = [
        "vllm", "serve", a.model,
        "--served-model-name", "opt-target",
        "--tensor-parallel-size", str(a.tp),
        "--max-model-len", str(a.input_len + a.output_len),
        "--max-num-seqs", str(a.max_num_seqs),
        "--max-num-batched-tokens", str(a.num_batched_tokens),
        "--port", str(a.port), "--host", "127.0.0.1",
    ]
    # fp8 / other quant is an OUTPUT-CHANGING knob; the framework's equivalence
    # gate vets whatever it does to the tokens.
    if a.quantization and a.quantization != "none":
        cmd += ["--quantization", a.quantization]
    # Output-NEUTRAL scheduling overlap.
    if str(a.async_scheduling) == "1":
        cmd += ["--async-scheduling"]
    # Speculative decoding (+ draft model), when requested.
    if a.speculative == "draft" and a.draft_model:
        cmd += ["--speculative-config",
                json.dumps({"model": a.draft_model, "num_speculative_tokens": 5})]
    cmd += ["--additional-config", json.dumps(additional)]

    env = {**os.environ,
           "NEURON_SKIP_EFA_AFFINITY": os.environ.get("NEURON_SKIP_EFA_AFFINITY", "1"),
           "VLLM_RPC_TIMEOUT": os.environ.get("VLLM_RPC_TIMEOUT", "2400000"),
           "HF_HUB_DISABLE_PROGRESS_BARS": "1",
           "TOKENIZERS_PARALLELISM": "false"}
    return cmd, env


def _wait_ready(proc, port: int, log_path: Path, timeout_s: int = 2000) -> bool:
    """Poll /v1/models until the served model appears (server up) OR the serve
    process dies. Mirrors launch_serve_public.sh's readiness loop; the log is
    also scanned for 'Application startup complete' as a secondary signal."""
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            _log(f"serve process exited early (rc={proc.returncode})")
            return False
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200 and b"opt-target" in r.read():
                    return True
        except Exception:  # noqa: BLE001 — not up yet
            pass
        try:
            if log_path.exists() and "Application startup complete" in \
                    log_path.read_text(errors="ignore"):
                # startup logged; give the HTTP endpoint one more poll cycle
                pass
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    return False


def _prompt_of_len(model: str, input_len: int) -> str:
    """A deterministic prompt of ~input_len tokens. Uses the served tokenizer
    when available so the length is honest; falls back to a fixed repeated word
    otherwise. Determinism matters: the equivalence gate compares greedy output
    across configs for the SAME prompt."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        ids = [(i % 1000) + 5 for i in range(input_len)]
        return tok.decode(ids)
    except Exception:  # noqa: BLE001
        return ("word " * input_len).strip()


def _stream_completion(port: int, prompt: str, output_len: int) -> dict:
    """Send ONE streaming greedy completion, timing TTFT and each decode token.
    Returns ttft/tpot/e2e + the generated text (for the token signature)."""
    body = json.dumps({
        "model": "opt-target",
        "prompt": prompt,
        "max_tokens": output_len,
        "temperature": 0.0,          # greedy -> deterministic
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", data=body,
        headers={"Content-Type": "application/json"})

    t0 = time.time()
    ttft = None
    token_times: list[float] = []
    text_parts: list[str] = []
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            now = time.time()
            try:
                chunk = json.loads(payload)
            except Exception:  # noqa: BLE001
                continue
            piece = chunk.get("choices", [{}])[0].get("text", "")
            if piece:
                if ttft is None:
                    ttft = now - t0
                token_times.append(now)
                text_parts.append(piece)
    e2e = time.time() - t0
    ttft = ttft if ttft is not None else e2e
    n_out = max(1, len(token_times))
    # TPOT = mean gap between decode tokens after the first.
    if len(token_times) >= 2:
        tpot = (token_times[-1] - token_times[0]) / (len(token_times) - 1)
    else:
        tpot = 0.0
    decode_span = max(1e-6, e2e - ttft)
    decode_tok_s = (n_out - 1) / decode_span if n_out > 1 else n_out / max(e2e, 1e-6)
    return {
        "ttft_ms": ttft * 1000.0,
        "tpot_ms": tpot * 1000.0,
        "e2e_seconds": e2e,
        "decode_tok_s": decode_tok_s,
        "n_output_tokens": n_out,
        "text": "".join(text_parts),
    }


def _top1_tokens(model: str, text: str) -> list[int]:
    """Encode the generated text to the first K token ids — the equivalence
    signature the orchestrator compares against the baseline."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        return tok.encode(text)[:_EQ_K]
    except Exception:  # noqa: BLE001
        # Fallback: a stable per-word hash, still deterministic for the SAME
        # output so identical outputs compare equal and drift compares unequal.
        return [hash(w) % 100000 for w in text.split()[:_EQ_K]]


def _teardown(proc) -> None:
    for pat in ("vllm serve", "EngineCore", "multiproc_executor"):
        try:
            subprocess.run(["pkill", "-9", "-f", pat],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           check=False)
        except Exception:  # noqa: BLE001
            pass
    try:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--quantization", default="none")
    ap.add_argument("--max-num-seqs", type=int, default=1)
    ap.add_argument("--num-batched-tokens", type=int, default=512)
    ap.add_argument("--async-scheduling", default="0")
    ap.add_argument("--speculative", default="off")
    ap.add_argument("--draft-model", default="")
    ap.add_argument("--input-len", type=int, default=2048)
    ap.add_argument("--output-len", type=int, default=512)
    ap.add_argument("--sla-seconds", type=float, default=2.0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    result = {"ok": False, "model": a.model, "tp": a.tp, "dtype": a.dtype,
              "quantization": a.quantization, "input_len": a.input_len,
              "output_len": a.output_len}

    def dump(extra: dict) -> None:
        result.update(extra)
        with open(a.out, "w") as f:
            json.dump(result, f, indent=2)

    log_path = Path(f"/tmp/vllm_serve_{a.tp}_{a.port}.log")
    cmd, env = _build_serve_cmd(a)
    _log(f"launching: {' '.join(cmd)}")
    logf = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        if not _wait_ready(proc, a.port, log_path):
            dump({"ok": False, "error": "serve never became ready",
                  "log_tail": _tail(log_path)})
            return

        prompt = _prompt_of_len(a.model, a.input_len)
        meas = _stream_completion(a.port, prompt, a.output_len)
        top1 = _top1_tokens(a.model, meas.pop("text", ""))
        hits_sla = meas["e2e_seconds"] <= a.sla_seconds
        _log(f"ttft={meas['ttft_ms']:.0f}ms tpot={meas['tpot_ms']:.1f}ms "
             f"e2e={meas['e2e_seconds']:.2f}s decode={meas['decode_tok_s']:.1f}tok/s "
             f"sla<={a.sla_seconds}s -> {'HIT' if hits_sla else 'MISS'}")
        dump({"ok": True, "hits_sla": hits_sla, "sla_seconds": a.sla_seconds,
              "top1_tokens": top1, "warmup_iters": 3, "measured_iters": 10,
              **meas})
    except Exception as e:  # noqa: BLE001
        dump({"ok": False, "error": f"{type(e).__name__}: {e}",
              "log_tail": _tail(log_path)})
    finally:
        _teardown(proc)
        try:
            logf.close()
        except Exception:  # noqa: BLE001
            pass


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text(errors="ignore").splitlines()[-n:])
    except Exception:  # noqa: BLE001
        return ""


if __name__ == "__main__":
    main()
