"""The orthogonal 2-D mesh must reproduce the unsharded model, for EVERY split.

The mesh MATH is proven in test_moe_mesh.py, but a wrong SCOPED COLLECTIVE (o_proj
reducing across EP columns, or the expert sum reducing across TP rows) cannot be caught
by inspecting weights -- it only shows up in the output. So this runs the real tiny
Qwen3.5-MoE through actual gloo process groups on CPU, sharding attention across the TP
row and experts across the EP column, and asserts every split's logits equal the
unsharded reference token-for-token.

world=4 covers pure TP (4,1), pure EP (1,4), and the genuinely mixed (2,2) -- which is
the case that only the orthogonal mesh can express. If the two axes' collectives ever
cross-talk, the (2,2) logits diverge and this fails.

Correctness gate, not a smoke test. The multiprocess work runs in _moe_2d_runner.py (a
real file, because torch's spawn re-execs the parent's __main__ by path and pytest's
__main__ is not a runnable file). It skips ONLY when a one-off gloo probe cannot start
(runner exit 42); once gloo runs, any error or divergence inside the mesh is a hard
failure (runner exit 1 with the child traceback), never swallowed into a skip.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

_SRC = os.path.dirname(os.path.abspath(__file__))
_RUNNER = os.path.join(_SRC, "_moe_2d_runner.py")


def _env():
    env = dict(os.environ)
    # children import `backends`; make the src dir importable regardless of how pytest
    # was invoked, and keep gloo on loopback.
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    return env


def _reference_logits():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_moe_2d_runner", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.reference_logits()


@pytest.fixture(scope="module")
def gloo():
    r = subprocess.run([sys.executable, _RUNNER, "probe"], env=_env(),
                       capture_output=True, text=True, timeout=300)
    if r.returncode == 42:
        pytest.skip("gloo multiprocess unavailable on this host")
    assert r.returncode == 0, f"probe crashed unexpectedly:\n{r.stderr}"


@pytest.mark.parametrize("tp,ep", [(4, 1), (2, 2), (1, 4)])
def test_every_2d_split_matches_the_unsharded_model(tp, ep, gloo, tmp_path):
    out_path = str(tmp_path / f"logits_{tp}x{ep}.pt")
    r = subprocess.run(
        [sys.executable, _RUNNER, "mesh", str(tp), str(ep), out_path],
        env=_env(), capture_output=True, text=True, timeout=600)
    # returncode 42 (env) is already filtered by the `gloo` fixture; anything nonzero
    # here is a genuine mesh failure -- fail loudly with the rank's traceback.
    assert r.returncode == 0, \
        f"tp={tp} ep={ep} mesh run failed (rc={r.returncode}):\n{r.stderr}"
    assert os.path.exists(out_path), f"rank 0 produced no logits:\n{r.stderr}"
    got = torch.load(out_path)
    want = _reference_logits()
    assert got.shape == want.shape
    assert torch.allclose(got, want, atol=1e-4, rtol=1e-4), \
        f"tp={tp} ep={ep} diverged: max|d|={float((got - want).abs().max()):.2e}"
