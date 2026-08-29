# Quickstart

The framework points at a HuggingFace model, optimizes it on Trainium2, verifies
correctness, and publishes a reproducible recipe. You can try the whole loop on a
laptop (no Neuron hardware) with the `mock` backend.

## 1. Install (once)

```bash
git clone https://github.com/arminagha1234/trainium-optimizer
cd trainium-optimizer
pip install -e .          # installs the `optimizer` CLI + minimal deps
```

This installs the `optimizer` package and three console commands:
`trainium-optimize`, `trainium-apply`, `trainium-measure`.

## 2. Run the full optimize → verify → publish loop (hardware-free demo)

```bash
python -m optimizer.run --backend mock
# or, equivalently:
trainium-optimize --backend mock
```

Runs end-to-end in ~90s and writes a leaderboard + recipe bundles under
`implementation/artifacts/`. (Results are synthetic on `mock` — for real numbers
use a Neuron backend; see [implementation/ENVIRONMENT.md](implementation/ENVIRONMENT.md).)

## 3. Reproduce a published recipe

Each optimized model has a bundle under `optimized_models/<slug>/` containing
`recipe.json`, `RECIPE.md`, and `reproduce.sh`. To re-run one:

```bash
cd optimized_models/qwen3-0-6b
python -m optimizer.apply     --backend native-pytorch-beta3   # apply the recipe's config + compile
python -m optimizer.measure   --backend native-pytorch-beta3   # measure, compare to the published tok/s
```

Both commands read `model_id` / `config` / the published metric from the
`recipe.json` in the current directory (CLI flags override it). Swap in
`--backend mock` to smoke-test the flow with no hardware (the number won't match
the native recipe — that's expected).

## Real (Neuron) backends

`--backend native-pytorch-beta3` (and `vllm-serve`) require the on-device Neuron
toolchain (torch-neuronx, neuronx-cc) on a trn2 instance. Setup — including which
pieces are public vs. Amazon-internal — is in
[implementation/ENVIRONMENT.md](implementation/ENVIRONMENT.md).

## Where things live

- `implementation/run_overnight.py` / `python -m optimizer.run` — the main loop
- `implementation/src/` — the framework (orchestrator, backends, opportunity sweep, kernel authoring, bank)
- `optimized_models/` — published recipe bundles
- `LEADERBOARD.md` — verified results index
- Deeper design docs: `architecture.md`, `optimization-stages.md`, `docs/`
