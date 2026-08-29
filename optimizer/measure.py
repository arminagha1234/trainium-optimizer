"""``python -m optimizer.measure`` — measure the throughput of a recipe's config
and check it lands within tolerance of the published number (the second step of
reproduce.sh).

Runs build_baseline -> apply_config(config) -> compile -> measure. When run from
a recipe bundle dir it reads model / backend / config / expected-metric from
``./recipe.json`` (so the generated ``reproduce.sh`` line — which passes no
``--set`` — works). ``--backend mock`` gives a hardware-free smoke test.
"""

from __future__ import annotations

import argparse

from optimizer._common import make_backend, resolve

# A small default shape sweep when --all-shapes is passed and the recipe doesn't
# pin its own. Shapes are seqlen strings (the backend's probe_shape contract).
_DEFAULT_SHAPES = ["512", "1024", "2048"]


def _shapes_from(recipe: dict | None, all_shapes: bool, shape: str | None) -> list[str]:
    if shape:
        return [shape]
    if recipe:
        ms = recipe.get("measurements")
        if isinstance(ms, list) and ms:
            got = [str(m.get("shape")) for m in ms if isinstance(m, dict) and m.get("shape")]
            if got:
                return got
    return _DEFAULT_SHAPES if all_shapes else ["1024"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="optimizer.measure", description=__doc__)
    ap.add_argument("--model", help="HF model id (or read from ./recipe.json)")
    ap.add_argument("--backend", help="mock | native-pytorch-beta3 | vllm-serve | diffusion-native")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument("--all-shapes", action="store_true")
    ap.add_argument("--shape", help="single seqlen to measure (overrides --all-shapes)")
    ap.add_argument("--tol", type=float, default=0.10, help="pass band vs published tok/s (frac)")
    a = ap.parse_args(argv)
    model, backend, config, recipe = resolve(a, need_config=True)
    batch = int(config.get("batch", 1) or 1)

    be = make_backend(backend)
    artifact = be.apply_config(be.build_baseline(model), config)
    neff = be.compile(artifact)

    shapes = _shapes_from(recipe, a.all_shapes, a.shape)
    best = 0.0
    print(f"[optimizer.measure] {model} on {backend} (batch={batch})")
    for shp in shapes:
        m = be.measure(neff, shp, batch)
        metric = float(getattr(m, "metric", 0.0) or 0.0)
        best = max(best, metric)
        print(f"  shape={shp:>6}  {metric:,.1f} {getattr(recipe,'get',lambda k,d=None:d)('metric_label') or 'tok/s'}")

    expected = float((recipe or {}).get("best_metric", 0.0) or 0.0)
    if expected > 0 and best > 0:
        ratio = best / expected
        ok = abs(ratio - 1.0) <= a.tol
        print(f"  best={best:,.1f} vs published {expected:,.1f} "
              f"({ratio*100:.0f}%) -> {'PASS' if ok else 'OUT OF TOLERANCE'} "
              f"(band ±{a.tol*100:.0f}%)")
        return 0 if ok else 1
    print(f"  best={best:,.1f} (no published metric to compare)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
