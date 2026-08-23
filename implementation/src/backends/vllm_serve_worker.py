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
import sys
import time
import urllib.request
from pathlib import Path

_EQ_K = 16  # equivalence signature length: first K greedy output token ids


def _log(msg: str) -> None:
    print(f"[vllm-serve-worker] {msg}", flush=True)


def _is_gemma4(model: str) -> bool:
    return "gemma" in str(model).lower()


def _vllm_bin() -> str:
    """Resolve the `vllm` CLI. Prefer the one next to the running interpreter
    (the venv that has vLLM-Neuron installed) so a subprocess launched with a
    bare PATH still finds it; fall back to PATH lookup."""
    import shutil
    cand = os.path.join(os.path.dirname(sys.executable), "vllm")
    if os.path.exists(cand):
        return cand
    return shutil.which("vllm") or "vllm"


# ── Tied-embedding lm_head fix ────────────────────────────────────────────────
# vLLM-Neuron's dense checkpoint loader (utils/checkpoints.py
# load_sharded_pipelined) is STRICT: the model always registers a separate
# `lm_head.weight` parameter and unconditionally maps it to the checkpoint key
# "lm_head.weight" (model/qwen3/model.py builds `mappings["lm_head.weight"] =
# "lm_head.weight"` with no tie branch). When a model TIES its embeddings
# (tie_word_embeddings=True) *and* the exported checkpoint does NOT materialize
# `lm_head.weight` (it shares model.embed_tokens.weight), the loader hits
#   RuntimeError: Checkpoint key(s) not found for parameter 'lm_head.weight' ...
# EngineCore init dies -> 0 tok/s -> the orchestrator reports FAIL_NO_BASELINE.
#
# NOTE the real differentiator (verified on-box): it is NOT tp / shard count.
# Qwen3-0.6B (tied, 1 file), Qwen3-1.7B (tied, 2 files) and Qwen3-8B (untied,
# 5 files) all PHYSICALLY contain lm_head.weight in their safetensors, so they
# load at any tp. Qwen3-4B is the one model that is tied AND ships no
# lm_head.weight (only embed_tokens.weight) -> it is the only one that fails,
# and it fails at EVERY tp. So constraining tp would not fix it.
#
# Fix (general, config-driven, additive): when the HF config says the model
# ties its embeddings and the resolved checkpoint has no materialized
# lm_head.weight, serve from a PATCHED LOCAL checkpoint directory — symlinks to
# every original file plus one extra safetensors carrying
#   lm_head.weight := model.embed_tokens.weight
# (exactly what "tied" means; identical [vocab, hidden] shape, so the
# ColumnParallelLinear lm_head shards it the same way at any tp). vLLM-Neuron's
# _LocalCheckpointSource enumerates *.safetensors in the dir and reads keys
# directly (no index.json dependency), so the added key is discovered and the
# strict loader is satisfied. Untied models, and tied models whose checkpoint
# already materializes lm_head (0.6B/1.7B/8B), are returned UNCHANGED. Every
# step is fail-open: any error falls back to serving the original model id, i.e.
# today's behavior.
_LMHEAD_KEY = "lm_head.weight"
_EMBED_KEY = "model.embed_tokens.weight"


def _hf_config_dict(model: str) -> dict:
    """The model's HF config as a dict. Empty (-> treated as untied) on any
    failure, so a config we cannot read never triggers the patch path."""
    try:
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(model, trust_remote_code=True).to_dict()
    except Exception:  # noqa: BLE001
        pass
    try:
        p = os.path.join(model, "config.json")
        if os.path.isfile(p):
            with open(p) as f:
                return json.load(f)
    except Exception:  # noqa: BLE001
        pass
    return {}


def _resolve_snapshot_dir(model: str) -> str | None:
    """Local directory holding the model's checkpoint files. A dir path is
    returned as-is; an HF id is resolved to its cached snapshot (fetched if the
    cache is cold). None on failure."""
    if os.path.isdir(model):
        return model
    try:
        from huggingface_hub import snapshot_download
        try:
            return snapshot_download(model, local_files_only=True)
        except Exception:  # noqa: BLE001 — not cached yet; allow a fetch
            return snapshot_download(model)
    except Exception:  # noqa: BLE001
        return None


def _checkpoint_has_lm_head(snapshot_dir: str) -> bool:
    """True if lm_head.weight is materialized in the checkpoint. Fail-SAFE True
    (assume present -> do NOT patch -> unchanged behavior) if we cannot tell."""
    try:
        import glob
        idx = glob.glob(os.path.join(snapshot_dir, "*.safetensors.index.json"))
        if idx:
            with open(idx[0]) as f:
                return _LMHEAD_KEY in json.load(f).get("weight_map", {})
        from safetensors import safe_open
        for fp in glob.glob(os.path.join(snapshot_dir, "*.safetensors")):
            with safe_open(fp, framework="pt") as sf:
                if _LMHEAD_KEY in sf.keys():
                    return True
        return False
    except Exception:  # noqa: BLE001
        return True


def _materialize_tied_lm_head(model: str, snapshot_dir: str) -> str | None:
    """Build a patched local checkpoint dir: symlinks to every file in
    snapshot_dir PLUS an added safetensors whose lm_head.weight aliases the tied
    model.embed_tokens.weight. Returns the patched dir, or None on failure."""
    try:
        import glob
        import hashlib
        from safetensors import safe_open
        from safetensors.torch import save_file
        emb = None
        for fp in glob.glob(os.path.join(snapshot_dir, "*.safetensors")):
            with safe_open(fp, framework="pt", device="cpu") as sf:
                if _EMBED_KEY in sf.keys():
                    emb = sf.get_tensor(_EMBED_KEY)
                    break
        if emb is None:
            return None
        tag = hashlib.sha1(os.path.abspath(snapshot_dir).encode()).hexdigest()[:12]
        patched = os.path.join(tempfile.gettempdir(), f"tied_lmhead_{tag}")
        os.makedirs(patched, exist_ok=True)
        for name in os.listdir(snapshot_dir):
            dst = os.path.join(patched, name)
            if not os.path.lexists(dst):
                try:
                    os.symlink(os.path.realpath(os.path.join(snapshot_dir, name)), dst)
                except FileExistsError:
                    pass
        lm_path = os.path.join(patched, "model-lmhead-tied.safetensors")
        if not os.path.exists(lm_path):
            save_file({_LMHEAD_KEY: emb.contiguous()}, lm_path)
        return patched
    except Exception:  # noqa: BLE001
        return None


def _resolve_served_model(model: str) -> str:
    """Return the path/id to hand `vllm serve`. For a tied-embedding model whose
    checkpoint has no materialized lm_head.weight, return a patched local
    checkpoint (see module note above) so the strict loader finds the key;
    otherwise return `model` unchanged. Fail-open on any error."""
    try:
        if not bool(_hf_config_dict(model).get("tie_word_embeddings", False)):
            return model
        snap = _resolve_snapshot_dir(model)
        if not snap or _checkpoint_has_lm_head(snap):
            return model
        patched = _materialize_tied_lm_head(model, snap)
        if patched:
            _log(f"tied-embedding model '{model}' ships no lm_head.weight; "
                 f"serving patched checkpoint {patched} "
                 f"(lm_head.weight := {_EMBED_KEY})")
            return patched
        return model
    except Exception:  # noqa: BLE001
        return model


def _build_serve_cmd(a) -> tuple[list[str], dict]:
    """Assemble the `vllm serve` argv + env, mirroring the proven recipe.

    Gemma-4-12B specifics (adapted from the on-box launch/bench scripts):
      - SINGLE-SHOT prefill: max-num-batched-tokens == max-model-len == the
        one num_batched_tokens bucket, so the whole prompt prefills in one shot
        (the gemma4_unified V2 prefill path requires this; chunked prefill is
        not supported by the impl).
      - on-device sampling (all_greedy) is a first-class LEVER here: with it ON
        the sampler runs on-device (no per-step host round-trip); with it OFF
        vLLM samples on the host (the round-trip is the host-bound decode tax we
        are measuring).
      - env: GEMMA4_V2_PREFILL=1 GEMMA4_BF16_FALLBACK=1 (the working recipe),
        plus generous Neuron/vLLM timeouts for the ~500s first-boot compile.
    """
    gemma = _is_gemma4(a.model)
    # Single-shot for Gemma: the prompt must fit one prefill bucket. The backend
    # passes num_batched_tokens == input+output (2560) for this; guard anyway.
    if gemma:
        max_model_len = a.input_len + a.output_len
        nbt = max(a.num_batched_tokens, max_model_len)  # never chunk the prompt
        nbt_bucket = nbt
    else:
        max_model_len = a.input_len + a.output_len
        nbt = a.num_batched_tokens
        nbt_bucket = a.num_batched_tokens

    neuron_config: dict = {
        "num_batched_tokens_buckets": [nbt_bucket],
        "num_seqs_buckets": [a.max_num_seqs],
    }
    # on-device sampling LEVER. ON -> all_greedy on device (default). OFF -> omit
    # the block so vLLM samples on the host (extra per-step round-trip).
    if str(a.on_device_sampling) == "1":
        neuron_config["on_device_sampling_config"] = {"all_greedy": True}
    additional = {"neuron_config": neuron_config}

    # Tied-embedding models with no materialized lm_head.weight (e.g. Qwen3-4B)
    # are served from a patched local checkpoint so vLLM-Neuron's strict loader
    # finds lm_head.weight; every other model resolves to itself unchanged.
    served_model = _resolve_served_model(a.model)
    cmd = [
        _vllm_bin(), "serve", served_model,
        "--served-model-name", "opt-target",
        "--tensor-parallel-size", str(a.tp),
        "--max-model-len", str(max_model_len),
        "--max-num-seqs", str(a.max_num_seqs),
        "--max-num-batched-tokens", str(nbt),
        "--no-enable-prefix-caching",
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

    # The venv bin dir MUST be on PATH so the serve subprocess can find the
    # `neuronx-cc` compiler binary (invoked by name during first-boot compile);
    # a bare inherited PATH lacks it and the serve dies with
    # "neuronx-cc compiler binary does not exist".
    _venv_bin = os.path.dirname(sys.executable)
    _path = os.environ.get("PATH", "")
    if _venv_bin and _venv_bin not in _path.split(os.pathsep):
        _path = _venv_bin + os.pathsep + _path
    env = {**os.environ,
           "PATH": _path,
           "NEURON_SKIP_EFA_AFFINITY": os.environ.get("NEURON_SKIP_EFA_AFFINITY", "1"),
           "VLLM_RPC_TIMEOUT": os.environ.get("VLLM_RPC_TIMEOUT", "3000000"),
           "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": os.environ.get("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "3000"),
           "VLLM_ENGINE_ITERATION_TIMEOUT_S": os.environ.get("VLLM_ENGINE_ITERATION_TIMEOUT_S", "3000"),
           "HF_HUB_DISABLE_PROGRESS_BARS": "1",
           "TOKENIZERS_PARALLELISM": "false"}
    if gemma:
        env.setdefault("GEMMA4_V2_PREFILL", "1")
        env.setdefault("GEMMA4_BF16_FALLBACK", "1")
        if str(a.on_device_sampling) == "1":
            env.setdefault("GEMMA4_V2_DECODE", os.environ.get("GEMMA4_V2_DECODE", "0"))
        # Per-config NEFF cache so tp/sampling variants don't collide but repeats
        # of the SAME graph reuse the compile.
        env.setdefault("VLLM_CACHE_ROOT",
                       f"/home/ubuntu/neff_sweep/tp{a.tp}_ods{a.on_device_sampling}_len{max_model_len}")
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


def _prompt_ids_of_len(model: str, input_len: int, bos_id: int) -> list[int]:
    """A deterministic prompt of EXACTLY input_len token ids, with BOS prepended.

    Gemma-4's tokenizer does NOT auto-add BOS, and the served completions
    endpoint won't add special tokens to a token-id prompt, so we prepend BOS
    (id 2) ourselves — exactly like the on-box bench harness
    (ids=[2] + repeated-text ...). Passing token ids (not a string) also means
    the input length is exact and identical across configs, which the greedy
    equivalence gate relies on."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        base = tok("The quick brown fox jumps over the lazy dog. ",
                   add_special_tokens=False)["input_ids"]
        ids = [bos_id] + (base * (input_len // len(base) + 1))[: input_len - 1]
        return ids
    except Exception:  # noqa: BLE001
        # Fallback: deterministic id range with BOS. Still exact length.
        return [bos_id] + [(i % 1000) + 5 for i in range(input_len - 1)]


def _stream_completion(port: int, prompt, output_len: int) -> dict:
    """Send ONE streaming greedy completion, timing TTFT and each decode token.
    `prompt` may be a string OR a list of token ids (Gemma path: BOS-prepended
    ids). Returns ttft/tpot/e2e + the generated text (for the token signature)."""
    body = json.dumps({
        "model": "opt-target",
        "prompt": prompt,
        "max_tokens": output_len,
        "min_tokens": output_len,    # force full OUT decode (SLA shape is fixed)
        "ignore_eos": True,
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
    ap.add_argument("--on-device-sampling", default="1")  # 1=on-device greedy, 0=host
    ap.add_argument("--bos-id", type=int, default=2)       # Gemma BOS
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
    proc = None
    try:
        proc = subprocess.Popen(cmd, env=env, stdout=logf,
                                stderr=subprocess.STDOUT, start_new_session=True)
        if not _wait_ready(proc, a.port, log_path):
            dump({"ok": False, "error": "serve never became ready",
                  "log_tail": _tail(log_path)})
            return

        prompt = _prompt_ids_of_len(a.model, a.input_len, a.bos_id)
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
        if proc is not None:
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
