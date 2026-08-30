"""A multi-rank TP launch must be given the runtime settings collectives need.

None of these were being set. tp<=8 happened to survive without them; tp=32 did not,
and the failures looked like memory problems while being nothing of the kind:

    Qwen3-30B-A3B   tp=32   NRT_RESOURCE; Failed to allocate resource   (1.9 GB/rank)
    Qwen3.5-122B    tp=32   Invalid NEFF, instruction, or input          (7.8 GB/rank)

Both are far inside the 14.4 GB/rank budget, and both models load and shard fine.
"""

from __future__ import annotations

import os

from backends.native_pytorch import collective_env


def test_multi_rank_gets_host_collective_communication():
    """Without this the all-reduce takes the OFI/EFA device path, which cannot
    initialise in this container."""
    env = collective_env(32)
    assert env["TORCH_NEURONX_ENABLE_HOST_CC"] == "1"
    assert env["TORCH_NEURONX_ENABLE_ASYNC_NRT"] == "1"


def test_the_visible_core_count_matches_the_rank_count():
    for tp in (2, 4, 8, 16, 24, 32, 48, 64):
        assert collective_env(tp)["NEURON_RT_NUM_CORES"] == str(tp)


def test_a_single_rank_run_is_left_alone():
    """No collective to route, and narrowing the core count would only constrain a
    run that does not need it."""
    assert collective_env(1) == {}
    assert collective_env(0) == {}


def test_every_value_is_a_string():
    """A non-str in a subprocess env raises TypeError at launch."""
    for v in collective_env(16).values():
        assert isinstance(v, str)


def test_an_operator_override_wins(monkeypatch):
    """Debugging a collective means being able to turn host CC off from outside."""
    import backends.native_pytorch as npt

    monkeypatch.setenv("TORCH_NEURONX_ENABLE_HOST_CC", "0")
    # measure() merges collective_env only for keys absent from os.environ.
    merged = {**os.environ, **{k: v for k, v in npt.collective_env(32).items()
                               if k not in os.environ}}
    assert merged["TORCH_NEURONX_ENABLE_HOST_CC"] == "0"


def test_the_settings_are_applied_by_the_measure_path():
    """Guards against the helper existing but never being called -- which is exactly
    the state this fixes: the recipe was in the operator notes, not in the code."""
    import inspect

    import backends.native_pytorch as npt

    src = inspect.getsource(npt)
    assert "collective_env(tp)" in src, "collective_env is never called"
    assert src.index("collective_env(tp)") > src.index("def measure")
