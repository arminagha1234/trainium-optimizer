"""``python -m optimizer.apply`` — apply a published recipe's config to a model
and compile it (the first step of reproduce.sh).

Runs the framework's real chain: build_baseline -> apply_config(config) ->
compile, on the requested backend. Config comes from ``--set k=v`` flags or, if
omitted, from ``./recipe.json``. Use ``--backend mock`` for a hardware-free
smoke test.
"""

from __future__ import annotations

import argparse
import time

from optimizer._common import make_backend, resolve


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="optimizer.apply", description=__doc__)
    ap.add_argument("--model", help="HF model id (or read from ./recipe.json)")
    ap.add_argument("--backend", help="mock | native-pytorch-beta3 | vllm-serve | diffusion-native")
    ap.add_argument("--set", action="append", default=[],
                    metavar="KEY=VAL", help="config override (repeatable)")
    a = ap.parse_args(argv)
    model, backend, config, _ = resolve(a, need_config=True)

    be = make_backend(backend)
    t0 = time.time()
    artifact = be.apply_config(be.build_baseline(model), config)
    neff = be.compile(artifact)
    compile_s = getattr(neff, "compile_seconds", time.time() - t0)
    print(f"[optimizer.apply] {model} on {backend}: applied {len(config)} config "
          f"key(s), compiled in {compile_s:.1f}s.")
    print(f"  config: {config}")
    print("  next: python -m optimizer.measure  (confirm throughput vs the recipe)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
