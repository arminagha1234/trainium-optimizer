"""P3 - CPU component-equivalence checks (runbook phase P3, section 8 subset).

CPU-only and checkpoint-independent (uses the vendored config.json), so it runs
in CI. Validates the transformers-native modeling implements the runbook's
required math; the full on-device equivalence ladder is in p2_device.py.
Skips cleanly if transformers deepseek_v4 modeling is unavailable.
"""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deepseek_v4.p1_reference import build_shrunk_config  # noqa: E402


def _model(cfg, seed=0):
    from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M
    torch.manual_seed(seed)
    return M.DeepseekV4ForCausalLM(cfg).float().eval()


def _ids(cfg, seq=32):
    return (torch.arange(seq).unsqueeze(0)) % cfg.vocab_size


def test_p1_reference_finite_and_reproducible():
    cfg = build_shrunk_config()
    ids = _ids(cfg)
    with torch.no_grad():
        l1 = _model(cfg)(ids).logits
        l2 = _model(cfg)(ids).logits
    assert torch.isfinite(l1).all()
    assert torch.equal(l1, l2)


def test_hc_sinkhorn_doubly_stochastic():
    # runbook 4.3 Critical Bug #1: the HyperConnection comb matrix must be
    # doubly-stochastic (row AND column sums ~ 1). Row-softmax-only (missing the
    # column normalisation) is the classic bug and would leave column sums off 1.
    cfg = build_shrunk_config(n_hash=4)          # all-hash: deterministic
    m = _model(cfg)
    hc = next(mod for _, mod in m.named_modules()
              if type(mod).__name__ == "DeepseekV4HyperConnection")
    grab = {}
    h = hc.register_forward_hook(lambda mod, i, o: grab.__setitem__("comb", o[1].detach()))
    with torch.no_grad():
        m(_ids(cfg))
    h.remove()
    comb = grab["comb"].float()                  # [B, S, hc_mult, hc_mult]
    row_err = (comb.sum(-1) - 1).abs().max().item()
    col_err = (comb.sum(-2) - 1).abs().max().item()
    assert col_err < 1e-3, f"comb columns not stochastic (Bug #1?): max|col-1|={col_err}"
    assert row_err < 5e-2, f"comb rows not stochastic: max|row-1|={row_err}"


def test_sec8_component_classes_present():
    m = _model(build_shrunk_config())
    names = {type(mod).__name__ for _, mod in m.named_modules()}
    for c in ["DeepseekV4RMSNorm", "DeepseekV4RotaryEmbedding", "DeepseekV4Attention",
              "DeepseekV4GroupedLinear", "DeepseekV4HyperConnection", "DeepseekV4HyperHead",
              "DeepseekV4TopKRouter", "DeepseekV4HashRouter", "DeepseekV4Experts", "DeepseekV4MLP"]:
        assert c in names, f"missing section-8 component {c}"
