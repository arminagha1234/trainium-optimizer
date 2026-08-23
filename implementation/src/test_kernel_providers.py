"""Tests for ``kernel_providers`` — the real LLM-provider adapters for the
kernel-author seam. All offline: no network, no real Bedrock/Anthropic call.

Coverage:
  * ``make_complete_fn("echo")`` round-trips through ``LLMAuthor`` into a
    well-formed ``AuthoredKernel``.
  * a monkeypatched boto3 ``bedrock-runtime`` client returns a fenced NKI block;
    ``bedrock_complete_fn`` extracts the text and ``LLMAuthor`` parses it.
  * ``make_complete_fn("bedrock")`` with boto3 absent raises a clear
    ``ProviderNotAvailable``.
  * the system preamble is present in the request body sent to the client.

Runnable two ways:
    python -m pytest -q test_kernel_providers.py
    python test_kernel_providers.py
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from invent_kernels import catalog
from kernel_author import LLMAuthor
from kernel_providers import (
    EmptyCompletion,
    KERNEL_AUTHORING_PREAMBLE,
    ProviderNotAvailable,
    anthropic_complete_fn,
    author_from_provider,
    bedrock_complete_fn,
    echo_complete_fn,
    make_complete_fn,
)


# ---------------------------------------------------------------------------
# fake boto3 — records the invoke_model body, returns a fenced NKI block
# ---------------------------------------------------------------------------
_FAKE_NKI = (
    "Here is the kernel:\n"
    "```python\n"
    "import nki\n"
    "import nki.isa as nisa\n"
    "import nki.language as nl\n"
    "@nki.jit\n"
    "def softcap_kernel(x, cap, out):\n"
    "    return out\n"
    "```\n"
    "That should compile.\n"
)


class _FakeBody:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _FakeBedrockClient:
    def __init__(self, sink: dict):
        self._sink = sink

    def invoke_model(self, *, modelId: str, body: str):  # noqa: N803 - boto3 kwarg name
        self._sink["modelId"] = modelId
        self._sink["body"] = json.loads(body)
        payload = {"content": [{"type": "text", "text": _FAKE_NKI}]}
        return {"body": _FakeBody(json.dumps(payload).encode("utf-8"))}


def _install_fake_boto3(monkeypatch, sink: dict):
    fake = types.ModuleType("boto3")

    def _client(service, region_name=None, **_):
        sink["service"] = service
        sink["region"] = region_name
        return _FakeBedrockClient(sink)

    fake.client = _client
    monkeypatch.setitem(sys.modules, "boto3", fake)


# ---------------------------------------------------------------------------
# echo provider round-trips through LLMAuthor
# ---------------------------------------------------------------------------
def test_echo_round_trips_through_llm_author():
    fn = make_complete_fn("echo")
    assert fn is echo_complete_fn
    author = LLMAuthor(fn)
    spec = catalog()["softcap"]
    kernel = author.author(spec, lessons=None, feedback=None)
    assert kernel.op == "softcap"
    assert kernel.entry == "softcap_kernel"       # entry matched the prompt's ask
    assert "@nki.jit" in kernel.nki_src
    assert "nl.arange" not in kernel.nki_src       # lint-clean stub


def test_author_from_provider_echo_builds_llm_author():
    author = author_from_provider("echo")
    assert isinstance(author, LLMAuthor)
    kernel = author.author(catalog()["attn_decode"], lessons=None, feedback=None)
    assert kernel.entry == "attn_decode_kernel"


# ---------------------------------------------------------------------------
# bedrock with a mocked client
# ---------------------------------------------------------------------------
def test_bedrock_complete_fn_extracts_text_and_llm_author_parses(monkeypatch):
    sink: dict = {}
    _install_fake_boto3(monkeypatch, sink)

    fn = bedrock_complete_fn(model_id="anthropic.claude-opus-5",
                             region="us-west-2", temperature=None)
    author = LLMAuthor(fn)
    spec = catalog()["softcap"]
    kernel = author.author(spec, lessons=None, feedback=None)

    # bedrock_complete_fn returned the model's fenced block; LLMAuthor parsed it.
    assert kernel.entry == "softcap_kernel"
    assert "def softcap_kernel(x, cap, out):" in kernel.nki_src
    # right service + region + model wired through.
    assert sink["service"] == "bedrock-runtime"
    assert sink["region"] == "us-west-2"
    assert sink["modelId"] == "anthropic.claude-opus-5"


def test_bedrock_request_carries_system_preamble_and_prompt(monkeypatch):
    sink: dict = {}
    _install_fake_boto3(monkeypatch, sink)

    fn = bedrock_complete_fn(temperature=None)
    author = LLMAuthor(fn)
    author.author(catalog()["softcap"], lessons=None, feedback=None)

    body = sink["body"]
    # the standing authoring contract rides the Messages `system` field...
    assert body["system"] == KERNEL_AUTHORING_PREAMBLE
    # ...and the per-round author prompt is the user message.
    assert body["messages"][0]["role"] == "user"
    assert "## Op to author" in body["messages"][0]["content"]
    assert "softcap_kernel" in body["messages"][0]["content"]
    assert body["anthropic_version"] == "bedrock-2023-05-31"


def test_bedrock_omits_temperature_when_none_includes_when_set(monkeypatch):
    sink: dict = {}
    _install_fake_boto3(monkeypatch, sink)

    # None -> not sent (current models reject sampling params with a 400).
    bedrock_complete_fn(temperature=None)("hi def x_kernel(")
    assert "temperature" not in sink["body"]

    # explicit value -> sent (older models still accept it).
    bedrock_complete_fn(temperature=0.0)("hi def x_kernel(")
    assert sink["body"]["temperature"] == 0.0


# ---------------------------------------------------------------------------
# BUG #2: thinking-token budget — stop_reason=max_tokens + no text must NOT
# silently return "" (empty authored source); it must raise EmptyCompletion.
# ---------------------------------------------------------------------------
class _MaxTokensBedrockClient:
    """Fake bedrock client that returns a thinking-only response: the model
    burned all of max_tokens on `thinking`, so stop_reason='max_tokens' and
    there is NO text block."""

    def __init__(self, sink: dict):
        self._sink = sink

    def invoke_model(self, *, modelId: str, body: str):  # noqa: N803
        self._sink["body"] = json.loads(body)
        payload = {
            "stop_reason": "max_tokens",
            "content": [{"type": "thinking", "thinking": "...long reasoning..."}],
        }
        return {"body": _FakeBody(json.dumps(payload).encode("utf-8"))}


def _install_fake_boto3_maxtokens(monkeypatch, sink: dict):
    fake = types.ModuleType("boto3")

    def _client(service, region_name=None, **_):
        return _MaxTokensBedrockClient(sink)

    fake.client = _client
    monkeypatch.setitem(sys.modules, "boto3", fake)


def test_bedrock_raises_on_max_tokens_with_no_text(monkeypatch):
    sink: dict = {}
    _install_fake_boto3_maxtokens(monkeypatch, sink)
    fn = bedrock_complete_fn(temperature=None)
    try:
        fn("author me a kernel: def softcap_kernel(")
    except EmptyCompletion as exc:
        # Diagnostic names the real cause and the fix (raise max_tokens).
        assert "max_tokens" in str(exc)
    else:
        raise AssertionError("expected EmptyCompletion, not a silent empty string")


def test_bedrock_default_max_tokens_raised_for_thinking_model():
    # BUG #2: the default must be well above the old 4096 so a thinking model has
    # room to emit an answer after its thinking block.
    import inspect
    default = inspect.signature(bedrock_complete_fn).parameters["max_tokens"].default
    assert default >= 16000, default


class _MaxTokensAnthropicResp:
    stop_reason = "max_tokens"
    content = [{"type": "thinking", "thinking": "...long reasoning..."}]


def _install_fake_anthropic_maxtokens(monkeypatch):
    fake = types.ModuleType("anthropic")

    class _Messages:
        def create(self, **_):
            return _MaxTokensAnthropicResp()

    class _Anthropic:
        def __init__(self, *a, **k):
            self.messages = _Messages()

    fake.Anthropic = _Anthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake)


def test_anthropic_raises_on_max_tokens_with_no_text(monkeypatch):
    _install_fake_anthropic_maxtokens(monkeypatch)
    fn = anthropic_complete_fn()
    try:
        fn("author me a kernel: def softcap_kernel(")
    except EmptyCompletion as exc:
        assert "max_tokens" in str(exc)
    else:
        raise AssertionError("expected EmptyCompletion, not a silent empty string")


def test_anthropic_default_max_tokens_raised_for_thinking_model():
    import inspect
    default = inspect.signature(anthropic_complete_fn).parameters["max_tokens"].default
    assert default >= 16000, default


# ---------------------------------------------------------------------------
# provider-not-available paths
# ---------------------------------------------------------------------------
def test_make_complete_fn_bedrock_without_boto3_raises(monkeypatch):
    # Force the lazy `import boto3` inside the factory to fail, regardless of
    # whether boto3 happens to be installed in the test environment.
    monkeypatch.setitem(sys.modules, "boto3", None)
    try:
        make_complete_fn("bedrock")
    except ProviderNotAvailable as exc:
        assert "boto3" in str(exc)
    else:
        raise AssertionError("expected ProviderNotAvailable when boto3 is absent")


def test_make_complete_fn_anthropic_without_sdk_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    try:
        make_complete_fn("anthropic")
    except ProviderNotAvailable as exc:
        assert "anthropic" in str(exc).lower()
    else:
        raise AssertionError("expected ProviderNotAvailable when anthropic SDK is absent")


def test_make_complete_fn_unknown_provider_raises():
    try:
        make_complete_fn("openai")
    except ValueError as exc:
        assert "unknown provider" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown provider")


def test_make_complete_fn_auto_raises_naming_both_remedies(monkeypatch):
    # No AWS creds resolvable, no ANTHROPIC_API_KEY -> a clear, actionable error.
    monkeypatch.setattr("kernel_providers._bedrock_resolvable", lambda: False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        make_complete_fn("auto")
    except ProviderNotAvailable as exc:
        msg = str(exc)
        assert "boto3" in msg and "ANTHROPIC_API_KEY" in msg
    else:
        raise AssertionError("expected ProviderNotAvailable when nothing resolves")


# ===========================================================================
# standalone runner (no pytest required)
# ===========================================================================
def _run_standalone() -> int:
    import inspect
    import tempfile
    import traceback

    class _MP:
        """Minimal monkeypatch shim so the suite runs without pytest."""

        def __init__(self):
            self._undo = []

        def setitem(self, d, k, v):
            missing = k not in d
            old = d.get(k)
            d[k] = v
            self._undo.append(lambda: (d.pop(k) if missing else d.__setitem__(k, old)))

        def setattr(self, target, value):
            mod_name, _, attr = target.rpartition(".")
            mod = sys.modules[mod_name]
            old = getattr(mod, attr)
            setattr(mod, attr, value)
            self._undo.append(lambda: setattr(mod, attr, old))

        def delenv(self, name, raising=True):
            import os as _os
            old = _os.environ.get(name)
            _os.environ.pop(name, None)
            self._undo.append(lambda: _os.environ.__setitem__(name, old) if old is not None else None)

        def undo(self):
            for fn in reversed(self._undo):
                fn()
            self._undo.clear()

    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in fns:
        params = inspect.signature(fn).parameters
        mp = _MP()
        try:
            kwargs = {}
            if "monkeypatch" in params:
                kwargs["monkeypatch"] = mp
            if "tmp_path" in params:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d), **kwargs)
            else:
                fn(**kwargs)
            print(f"  PASS  {name}")
            passed += 1
        except Exception:  # noqa: BLE001
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
        finally:
            mp.undo()
    print(f"\n{passed} passed, {failed} failed (of {len(fns)})")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
