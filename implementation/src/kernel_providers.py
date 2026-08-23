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
    "bf16-in/fp32-accumulate. Apply the compiler-error fix provided."
)

# Bedrock's InvokeModel path pins the Anthropic schema by version string.
_BEDROCK_ANTHROPIC_VERSION = "bedrock-2023-05-31"


class ProviderNotAvailable(RuntimeError):
    """Raised when a requested provider cannot be constructed — the message
    always names the concrete thing the caller must install or set."""


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


# ---------------------------------------------------------------------------
# Bedrock (primary)
# ---------------------------------------------------------------------------
def bedrock_complete_fn(model_id: str = "anthropic.claude-opus-5",
                        region: str | None = None,
                        max_tokens: int = 4096,
                        temperature: float | None = 0.0) -> CompleteFn:
    """Return a ``complete_fn`` backed by boto3 ``bedrock-runtime`` InvokeModel.

    Uses the Anthropic Messages request/response schema. ``region`` falls back
    to ``$AWS_REGION`` then ``$AWS_DEFAULT_REGION``; credentials resolve through
    the standard boto3 chain (env, shared config, or — on the trn2 EC2 boxes —
    the instance role). boto3 is imported here, not at module import, so this
    module loads without boto3 installed and a missing dependency surfaces as a
    clear ``ProviderNotAvailable`` at factory-call time.

    Note on ``temperature``: current Anthropic models reject sampling params
    (400). Pass ``temperature=None`` when the resolved ``model_id`` is a
    current-generation model; the default ``0.0`` suits older models where a
    deterministic sample is still accepted.
    """
    try:
        import boto3  # noqa: PLC0415 — lazy so the module imports without boto3
    except ImportError as exc:  # pragma: no cover - exercised via make_complete_fn test
        raise ProviderNotAvailable(
            "Bedrock provider requires boto3. Install it (`pip install boto3`) "
            "or select provider='anthropic' with ANTHROPIC_API_KEY set."
        ) from exc

    resolved_region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    client = boto3.client("bedrock-runtime", region_name=resolved_region)

    def _complete(prompt: str) -> str:
        body = _messages_body(prompt, max_tokens, temperature, include_version=True)
        resp = client.invoke_model(modelId=model_id, body=json.dumps(body))
        raw = resp["body"].read()
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
        return _text_from_content(payload.get("content", []))

    return _complete


# ---------------------------------------------------------------------------
# Anthropic SDK (fallback)
# ---------------------------------------------------------------------------
def anthropic_complete_fn(model: str = "claude-opus-5",
                          max_tokens: int = 4096) -> CompleteFn:
    """Return a ``complete_fn`` backed by the ``anthropic`` SDK.

    Reads ``ANTHROPIC_API_KEY`` from the environment (the SDK's default client
    resolution). Lazy-imported so this module loads without the SDK; a missing
    dependency surfaces as a clear ``ProviderNotAvailable``."""
    try:
        import anthropic  # noqa: PLC0415 — lazy so the module imports without anthropic
    except ImportError as exc:  # pragma: no cover - exercised via make_complete_fn test
        raise ProviderNotAvailable(
            "Anthropic provider requires the `anthropic` SDK. Install it "
            "(`pip install anthropic`) and set ANTHROPIC_API_KEY, or select "
            "provider='bedrock'."
        ) from exc

    client = anthropic.Anthropic()

    def _complete(prompt: str) -> str:
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=KERNEL_AUTHORING_PREAMBLE,
            messages=[{"role": "user", "content": prompt}],
        )
        return _text_from_content(resp.content)

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
