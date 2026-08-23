"""kernel_providers.py — real LLM-provider adapters for the kernel-author seam.

``kernel_author.LLMAuthor`` authors kernels through an INJECTED
``complete_fn(prompt: str) -> str``. That is the whole provider seam: everything
provider-specific lives behind that one callable, and ``LLMAuthor`` /
``build_author_prompt`` / ``extract_nki_source`` never know which model answered.

This module supplies production ``complete_fn`` implementations so the repair
loop can author kernels against a real model, plus the glue to pick one from the
environment:

  * ``bedrock_complete_fn``   — boto3 ``bedrock-runtime`` ``invoke_model`` in the
    Anthropic Messages format. This is the PRIMARY target: the trn2 boxes are
    EC2 instances whose instance role may carry Bedrock access, so no API key
    has to be shipped. boto3 is lazy-imported inside the factory, so importing
    this module never requires boto3.
  * ``anthropic_complete_fn`` — the ``anthropic`` SDK (reads ``ANTHROPIC_API_KEY``),
    also lazy-imported. The fallback for a laptop with a key but no AWS creds.
  * ``echo_complete_fn``      — deterministic, fully offline; emits a lint-clean
    ``@nki.jit`` stub keyed off the entry the prompt asks for. For tests / CI.
  * ``make_complete_fn``      — provider picker (``auto`` / ``bedrock`` /
    ``anthropic`` / ``echo``) that raises a clear, actionable error naming what
    to set when nothing is resolvable.
  * ``author_from_provider``  — one-liner returning ``LLMAuthor(make_complete_fn(...))``.

The system-prompt preamble (``KERNEL_AUTHORING_PREAMBLE``) is sent as the
Messages ``system`` field on every real request, so the model's standing
contract (NKI, static shapes, output-tensors-as-args, nc_matmul contraction,
bf16-in/fp32-accumulate, apply the fed-back fix) is stated once, out of band
from the per-round authoring prompt that ``build_author_prompt`` produces.
"""

from __future__ import annotations

import json
import os
import re

from kernel_author import CompleteFn, LLMAuthor

# ---------------------------------------------------------------------------
# system-prompt preamble — the standing authoring contract
# ---------------------------------------------------------------------------
# Prepended (as the Messages `system` field) to the per-round prompt that
# `build_author_prompt` assembles. Kept concise on purpose: the mandatory NKI
# lint rules and the fed-back compiler fix already live in the user prompt.
KERNEL_AUTHORING_PREAMBLE = (
    "You write NKI kernels for AWS Trainium. Output ONLY a fenced python code "
    "block with an @nki.jit kernel. Contract: static shapes, output tensors as "
    "kernel args, contraction on the partition axis via nisa.nc_matmul, "
    "bf16-in/fp32-accumulate. Apply the compiler-error fix provided. "
    "NKI 0.6.0 gen3/trn2 (from on-device compiler errors): nc_matmul RETURNS "
    "the result tile (no dst=/out= param) — call `psum = nisa.nc_matmul("
    "stationary, moving)`; the moving free dim must be <=512 (tile larger dims "
    "and accumulate in PSUM), stationary M<=128, contraction<=128; keep "
    "reductions 2-D (`nl.sum(x, axis=1, keepdims=True)`, never 1-D); use "
    "`nl.broadcast_to(tile, shape)` not the tensor method; multiply by scalars "
    "with a matching-dtype tile via nl.multiply, not a bare python float. "
    # PERFORMANCE RULES — mirror of kernel_author._PERF_PREAMBLE. A correct-but-
    # slow kernel is banked as an anti_pattern (a loss), so write for speed from
    # the first draft.
    "PERFORMANCE RULES (write for speed from the first draft — a correct-but-slow "
    "kernel is a loss): (1) FUSE the whole op into one kernel — intermediates "
    "stay in SBUF, one load per input and one store, never round-trip HBM (no HW "
    "cache). (2) FUSE onto the Scalar engine via nisa.activation(op=, bias=, "
    "scale=, reduce_op=nl.add) = op(scale*x+bias)+free-axis-reduce in ONE "
    "instruction (rmsnorm: op=square+reduce=add for mean-square in one pass, "
    "don't materialize a squared tile then sum; softmax: op=exp with -row-max as "
    "bias + reduce=add; softcap: tanh via scale=1/cap). (3) HOIST loop-invariant "
    "loads (gamma/beta/cap) OUT of the tile loop — no HW cache, don't re-DMA per "
    "tile. (4) KEEP THE PE BUSY — route reduce->Scalar and apply->Vector so they "
    "overlap, broadcast gamma via a TensorE matmul-against-ones (3-engine "
    "pipeline, not one serial engine). (5) bf16-in/fp32-accumulate, cast at the "
    "final store. (6) wide aligned tiles (partition=128, free>=512, bf16>=1024). "
    "(7) double-buffer so tile n+1's load overlaps tile n's compute (latency = "
    "max(compute,dma)). (8) reductions stay 2-D keepdims, delay the division "
    "(apply 1/sum via nisa.reciprocal to the final result)."
)

# Bedrock's InvokeModel path pins the Anthropic schema by version string.
_BEDROCK_ANTHROPIC_VERSION = "bedrock-2023-05-31"


class ProviderNotAvailable(RuntimeError):
    """Raised when a requested provider cannot be constructed — the message
    always names the concrete thing the caller must install or set."""


class EmptyCompletion(RuntimeError):
    """Raised when a model returns ``stop_reason == "max_tokens"`` with NO text
    block — the response is all ``thinking`` and no answer.

    BUG #2: Opus-5 is a thinking model. With ``max_tokens`` too small the model
    can spend the ENTIRE budget on the ``thinking`` block and return zero
    ``text`` blocks with ``stop_reason == "max_tokens"``. The old code silently
    returned ``""`` from ``_text_from_content``, so ``LLMAuthor`` got an empty
    authored source and the round failed with a misleading "no source" symptom
    rather than the real cause. Surface it loudly and name the fix instead."""


# ---------------------------------------------------------------------------
# offline / test provider
# ---------------------------------------------------------------------------
_ENTRY_IN_PROMPT_RE = re.compile(r"\bdef\s+(\w+_kernel)\b")


def echo_complete_fn(prompt: str) -> str:
    """Deterministic, offline ``complete_fn``.

    Emits a single lint-clean ``@nki.jit`` stub whose entry name matches the
    ``<op>_kernel`` the prompt asks for (``build_author_prompt`` writes an
    ``entry naming: define ``@nki.jit def <op>_kernel(...)``` line), so the
    result round-trips through ``extract_nki_source`` / ``extract_entry`` into a
    well-formed ``AuthoredKernel``. No network, no randomness — safe for CI."""
    m = _ENTRY_IN_PROMPT_RE.search(prompt or "")
    entry = m.group(1) if m else "echo_kernel"
    return (
        "```python\n"
        "import nki\n"
        "import nki.isa as nisa\n"
        "import nki.language as nl\n"
        "\n"
        "@nki.jit\n"
        f"def {entry}(*args):\n"
        "    # deterministic offline stub — real math is decided on device\n"
        "    return args[-1]\n"
        "```"
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _messages_body(prompt: str, max_tokens: int, temperature: float | None,
                   *, include_version: bool) -> dict:
    """Build the Anthropic Messages request body shared by Bedrock InvokeModel
    and (structurally) the anthropic SDK call.

    ``temperature`` is included ONLY when not None: current Anthropic models
    (Opus 5 / 4.8 / 4.7, Sonnet 5, Fable 5) reject any sampling parameter with a
    400, so pass ``temperature=None`` (the recommended setting) when targeting
    them and rely on prompting for determinism. It is honored on older models."""
    body: dict = {
        "max_tokens": max_tokens,
        "system": KERNEL_AUTHORING_PREAMBLE,
        "messages": [{"role": "user", "content": prompt}],
    }
    if include_version:
        body["anthropic_version"] = _BEDROCK_ANTHROPIC_VERSION
    if temperature is not None:
        body["temperature"] = temperature
    return body


def _text_from_content(content: list) -> str:
    """Concatenate the text of every ``text`` block in an Anthropic Messages
    ``content`` array (list of dicts or SDK block objects)."""
    parts: list[str] = []
    for block in content or []:
        btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
        if btype == "text":
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", "")
            if text:
                parts.append(text)
    return "".join(parts)


def _guard_empty_completion(text: str, stop_reason, max_tokens: int) -> str:
    """Return ``text`` unless the model hit the token cap before emitting any —
    then raise ``EmptyCompletion`` naming the fix (raise ``max_tokens``).

    BUG #2 guard: ``stop_reason == "max_tokens"`` with an empty ``text`` means a
    thinking model burned the whole budget on ``thinking`` and produced no
    answer. Returning ``""`` here would hand ``LLMAuthor`` an empty source and
    mask the cause; raising makes the real problem — and its remedy — explicit."""
    if not text and stop_reason == "max_tokens":
        raise EmptyCompletion(
            f"model returned stop_reason='max_tokens' with no text block: the "
            f"thinking budget consumed all {max_tokens} tokens before any answer "
            f"was written. Raise max_tokens (thinking models spend tokens on the "
            f"`thinking` block before the `text` block)."
        )
    return text


# ---------------------------------------------------------------------------
# Bedrock (primary)
# ---------------------------------------------------------------------------
def bedrock_complete_fn(model_id: str = "anthropic.claude-opus-5",
                        region: str | None = None,
                        max_tokens: int = 32000,
                        temperature: float | None = 0.0) -> CompleteFn:
    """Return a ``complete_fn`` backed by boto3 ``bedrock-runtime`` InvokeModel.

    Uses the Anthropic Messages request/response schema. ``region`` falls back
    to ``$AWS_REGION`` then ``$AWS_DEFAULT_REGION``, and a clear
    ``ProviderNotAvailable`` is raised if none resolves (boto3 would otherwise
    fail deep inside the first call with an opaque region error). Credentials
    resolve through the standard boto3 chain (env, shared config, or — on the
    trn2 EC2 boxes — the instance role). boto3 is imported here, not at module
    import, so this module loads without boto3 installed and a missing
    dependency surfaces as a clear ``ProviderNotAvailable`` at factory-call time.

    The client is built with an explicit ``botocore`` ``Config``: a 600s
    ``read_timeout`` (Opus-5's thinking pass routinely runs well past boto3's 60s
    default read timeout — a small default caused mid-authoring read timeouts),
    a 15s ``connect_timeout``, and 3 retry attempts.

    Note on ``temperature``: current Anthropic models reject sampling params
    (400). Pass ``temperature=None`` when the resolved ``model_id`` is a
    current-generation model; the default ``0.0`` suits older models where a
    deterministic sample is still accepted.

    ``max_tokens`` defaults to 32000 (BUG #2): Opus-5 is a thinking model and a
    small cap (the old 4096) can be exhausted entirely by the ``thinking`` block
    — a thinking model can spend 16k on thinking alone before writing any answer
    — leaving zero ``text`` blocks. A response that still comes back with
    ``stop_reason == "max_tokens"`` and no text raises ``EmptyCompletion`` rather
    than silently returning an empty authored source.
    """
    try:
        import boto3  # noqa: PLC0415 — lazy so the module imports without boto3
        from botocore.config import Config as _BotoConfig  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised via make_complete_fn test
        raise ProviderNotAvailable(
            "Bedrock provider requires boto3. Install it (`pip install boto3`) "
            "or select provider='anthropic' with ANTHROPIC_API_KEY set."
        ) from exc

    resolved_region = (region or os.environ.get("AWS_REGION")
                       or os.environ.get("AWS_DEFAULT_REGION"))
    if not resolved_region:
        raise ProviderNotAvailable(
            "Bedrock provider needs an AWS region: pass region=... or set "
            "AWS_REGION (or AWS_DEFAULT_REGION) in the environment."
        )
    # Long read_timeout: Opus-5's thinking pass exceeds boto3's 60s default and
    # was timing out mid-authoring. connect_timeout + bounded retries keep a
    # transient network blip from failing the whole round.
    boto_config = _BotoConfig(read_timeout=600, connect_timeout=15,
                              retries={"max_attempts": 3})
    client = boto3.client("bedrock-runtime", region_name=resolved_region,
                          config=boto_config)

    def _complete(prompt: str) -> str:
        body = _messages_body(prompt, max_tokens, temperature, include_version=True)
        resp = client.invoke_model(modelId=model_id, body=json.dumps(body))
        raw = resp["body"].read()
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
        text = _text_from_content(payload.get("content", []))
        return _guard_empty_completion(text, payload.get("stop_reason"), max_tokens)

    return _complete


# ---------------------------------------------------------------------------
# Anthropic SDK (fallback)
# ---------------------------------------------------------------------------
def anthropic_complete_fn(model: str = "claude-opus-5",
                          max_tokens: int = 32000) -> CompleteFn:
    """Return a ``complete_fn`` backed by the ``anthropic`` SDK.

    Reads ``ANTHROPIC_API_KEY`` from the environment (the SDK's default client
    resolution). Lazy-imported so this module loads without the SDK; a missing
    dependency surfaces as a clear ``ProviderNotAvailable``.

    ``max_tokens`` defaults to 32000 for the same reason as ``bedrock_complete_fn``
    (BUG #2): a thinking model can spend 16k on ``thinking`` alone before writing
    any answer. A response with ``stop_reason == "max_tokens"`` and no text raises
    ``EmptyCompletion`` instead of returning an empty authored source.

    The client is constructed with a 600s ``timeout`` and 3 ``max_retries`` (same
    spirit as the Bedrock read-timeout fix — a thinking model's long generation
    exceeds the SDK's short default) when the installed SDK accepts those kwargs;
    it degrades to the default client if it does not."""
    try:
        import anthropic  # noqa: PLC0415 — lazy so the module imports without anthropic
    except ImportError as exc:  # pragma: no cover - exercised via make_complete_fn test
        raise ProviderNotAvailable(
            "Anthropic provider requires the `anthropic` SDK. Install it "
            "(`pip install anthropic`) and set ANTHROPIC_API_KEY, or select "
            "provider='bedrock'."
        ) from exc

    try:
        client = anthropic.Anthropic(timeout=600.0, max_retries=3)
    except TypeError:  # pragma: no cover - very old SDK without these kwargs
        client = anthropic.Anthropic()

    def _complete(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=KERNEL_AUTHORING_PREAMBLE,
            messages=[{"role": "user", "content": prompt}],
        )
        text = _text_from_content(resp.content)
        return _guard_empty_completion(text, getattr(resp, "stop_reason", None), max_tokens)

    return _complete


# ---------------------------------------------------------------------------
# provider picker
# ---------------------------------------------------------------------------
def _bedrock_resolvable() -> bool:
    """True if boto3 imports AND some AWS credential resolves (env, shared
    config, or instance role) — the two things Bedrock InvokeModel needs."""
    try:
        import botocore.session  # noqa: PLC0415
    except ImportError:
        return False
    try:
        return botocore.session.get_session().get_credentials() is not None
    except Exception:  # noqa: BLE001 - any credential-resolution error => not resolvable
        return False


def make_complete_fn(provider: str = "auto", **kw) -> CompleteFn:
    """Return a production ``complete_fn`` for the chosen provider.

    ``provider``:
      * ``"bedrock"``   — boto3 InvokeModel (raises ``ProviderNotAvailable`` if
        boto3 is absent).
      * ``"anthropic"`` — anthropic SDK (raises ``ProviderNotAvailable`` if the
        SDK is absent).
      * ``"echo"``      — deterministic offline stub (tests / CI).
      * ``"auto"``      — Bedrock if boto3 + resolvable creds, else Anthropic if
        ``ANTHROPIC_API_KEY`` is set, else raise naming both remedies.

    Extra kwargs pass through to the selected factory (``model_id``, ``region``,
    ``max_tokens``, ``temperature`` for bedrock; ``model``, ``max_tokens`` for
    anthropic). ``echo`` takes no kwargs.
    """
    provider = (provider or "auto").lower()

    if provider == "echo":
        return echo_complete_fn
    if provider == "bedrock":
        return bedrock_complete_fn(**kw)
    if provider == "anthropic":
        return anthropic_complete_fn(**kw)
    if provider == "auto":
        if _bedrock_resolvable():
            return bedrock_complete_fn(**{k: v for k, v in kw.items()
                                          if k in ("model_id", "region", "max_tokens", "temperature")})
        if os.environ.get("ANTHROPIC_API_KEY"):
            return anthropic_complete_fn(**{k: v for k, v in kw.items()
                                            if k in ("model", "max_tokens")})
        raise ProviderNotAvailable(
            "No LLM provider resolvable for provider='auto'. To use Bedrock: "
            "install boto3 and provide AWS credentials (env vars or an EC2 "
            "instance role) plus AWS_REGION. To use Anthropic: `pip install "
            "anthropic` and set ANTHROPIC_API_KEY. For offline tests/CI: "
            "make_complete_fn('echo')."
        )
    raise ValueError(
        f"unknown provider {provider!r}; expected one of "
        "'auto', 'bedrock', 'anthropic', 'echo'"
    )


def author_from_provider(provider: str = "auto", **kw) -> LLMAuthor:
    """Convenience: ``LLMAuthor(make_complete_fn(provider, **kw))`` — a
    provider-backed author the engine can drop straight into its repair loop."""
    return LLMAuthor(make_complete_fn(provider, **kw))
