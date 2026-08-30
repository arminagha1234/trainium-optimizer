<!-- Consolidated into main from the `trn2-48xl-kaizen-sweep` branch so this
     record is not split across branches. The branch was deleted; the only other
     files it held were the pre-#111 copies of implementation/src/kernels/**, which
     that PR deliberately moved to knowledge-bank/kernels/. -->

> ## STATUS: numbers RETRACTED, findings mostly SUPERSEDED
>
> Kept as a record of the first 48xl run and of how its conclusions were reached.
> **Do not cite the GatedDeltaNet numbers below.**
>
> **Retraction.** Every GatedDeltaNet row here was measured against repo state
> `bd70a518` (#82). `280e3542` (#96) -- the device fast-path in
> `build_gdn_forward`, worth ~1250x on the serving path -- landed hours after both
> sweeps cloned, so the DeltaNet adapter was still on the numpy sim/recompile
> route. Invalidated:
>
> - Qwen3.5-0.8B / 2B / 4B speedups
> - **Qwen3.8-27B's 340 tok/s baseline**
> - `metric=0` for every `tp_degree` and `batch` config
> - Qwen3.5-9B `FAIL_NO_BASELINE`
>
> Qwen3-0.6B is unaffected -- dense, never touches the DeltaNet adapter -- so its
> 10.35x on this box stands.
>
> **What has since been fixed, and where.** Several findings below describe
> limitations that no longer exist:
>
> | Finding here | Status |
> |:--|:--|
> | `max_tp = 4` binds 27B-class Qwen3.5 | fixed, #121 raised the cap to the head count |
> | experts replicated on every rank | fixed, #125 expert parallelism |
> | GQA KV heads sliced to zero at tp>nkv | fixed, #127 |
> | DeltaNet key heads sliced to zero at tp>nkv | fixed, #135 |
> | "batch/tp_degree return no throughput on 48xl" | retracted AND separately explained: a timed-out candidate stranded its ranks and wedged the box (#131) |
> | `publish.py` fails without xattrs | fixed on the same branch, folded into main |
>
> The still-current guidance from this sweep lives in
> `docs/large-model-playbook.md`; the compiler crash it hints at is tracked in
> `docs/compiler-bugs-observed.md` and issue #134.

# trn2.48xlarge sweep on Kaizen — Qwen3.5 / 3.6 / 3.8 (2026-08-28)

First run of this framework on a **full trn2.48xlarge** (64 NeuronCores, LNC=2,
~1.5 TB HBM), obtained through **Kaizen** shared capacity rather than a Capacity
Block. Backend `native-pytorch-beta3` in the Beta 3 native-PyTorch DLC.

> **Do not merge these rows into `LEADERBOARD.md`.** That file is auto-published by
> the optimizer loop, and every row in it was measured on **trn2.3xlarge**. The
> numbers below are on **trn2.48xlarge** with a `--max-configs` backstop of 2–3, so
> they are *not* comparable to the published standings. See "Why 10.35x here vs
> 28.109x published" below — the discrepancy is a real finding, not a regression.

## Verified results

| Model | Arch | Params | Baseline (tok/s) | MFU | Best found | Speedup | State |
|:------|:-----|-------:|-----------------:|----:|-----------:|--------:|:------|
| Qwen3-0.6B | dense | 0.6B | 3,473 | 1.27% → 13.10% | **35,936** | **10.35×** | complete, grader-verified |
| Qwen3.5-0.8B | hybrid GatedDeltaNet | 0.8B | 1,097 | 0.50% | none kept | — | truncated at 90-min cap |
| Qwen3.8-27B | hybrid GatedDeltaNet | 27B | **340** | 1.18% | none in Stage 1 | — | baseline verified; search in progress |

- **Qwen3-0.6B** — win was `compile_mode=compile-default` (56.6 s compile), 22
  attempts. Trusted grader remeasured 36,050 tok/s (0.3% drift), `equivalence=ok`,
  correctness 100%. Box throughput 428,752 tok/s across 12/12 replicas at tp1.
- **Qwen3.5-0.8B** — best config was `cp_degree=8` at 1,106 tok/s, only **+0.75%**
  over baseline, correctly discarded as noise. All `tp_degree` and `batch` configs
  returned no throughput.
- **Qwen3.8-27B** — **the 30B-class hybrid does establish a baseline on trn2.48xl at
  tp=4** (6 min to baseline). Stage 1 found no improvement: `attn_implementation=sdpa`
  reached 351.7 tok/s but at 81.25% correctness and OOM at 95% HBM; `cp_degree=2/4/8`
  held 100% correctness at ~345 tok/s but also OOM'd at 95% HBM; every `tp_degree`
  and `batch` config returned `metric=0`.

## Findings

### 1. Linear-attention models are gated behind two switches, not one
Every Qwen3.5/3.6/3.8 model pre-flight skips with:

```
PRE-FLIGHT SKIP (linear-attention/GatedDeltaNet -> needs the DeltaNet kernel
(none registered on this install; the naive graph ISA-fails neuronx-cc).
Point $TRN_OPT_KERNEL_DIR at a DeltaNet kernel to optimize this model.)
```

Fixing it requires **both**:

```bash
export TRN_OPT_KERNEL_DIR=<repo>/implementation/src/kernels
python run_overnight.py ... --kernels-wired      # defaults to OFF
```

Setting only the env var is not enough. Worth noting the skip exits **0**, so a CI
job that trusts exit codes will report success on a run that did no work.

### 2. `max_tp = 4` is the binding constraint on 27B-class Qwen3.5 models
`native_pytorch.py::_fit_baseline_tp` caps TP at 4 for `Qwen3_5`/`Gemma4`
architectures. A 27B is ~67 GB bf16 → ~17 GB/rank at tp4, above the "keep weights
under ~10 GB/rank" target in that same function. The result is 95% HBM occupancy:
the baseline fits, but any config that adds memory pressure OOMs. The usual escape
(shard wider) is unavailable because tp≥8 is capped out, and all tp≥8 attempts
returned no throughput.

Arithmetically tp=8 looks viable for Qwen3.8-27B (`num_attention_heads=24`,
`linear_num_key_heads=16`, both divisible by 8), so the cap appears to be a
*validation* limit rather than a hard one. Raising it to 8 for Qwen3.5 is the
highest-value next experiment for this size class.

### 3. Why 10.35x here vs 28.109x published
The published Qwen3-0.6B row (28.109×) won with `batch=8` on trn2.3xlarge. On
trn2.48xlarge, `batch=8` and `batch=32` both returned `metric=0 -> backend produced
no throughput`, as did every `tp_degree` config. The winning lever on the small box
is unavailable on the big box, so the search falls back to `compile_mode` alone.
**The batch and tp config paths appear to be broken specifically on trn2.48xlarge
(LNC=2, 64-core).** This is the single most valuable thing to chase from this sweep.

### 4. Architecture classification corrections
- All of Qwen3.5-0.8B/2B/4B/9B/27B, Qwen3.6-27B and Qwen3.8-27B are
  **hybrid linear-attention** (`layer_types` interleaves `linear_attention` and
  `full_attention`; `full_attention_interval=4`). `model-landscape.md` lists
  Qwen3.6-27B as "27B dense" — that is incorrect.
- All report `Qwen3_5ForConditionalGeneration` / `model_type: qwen3_5`, and all carry
  a `vision_config` (multimodal wrappers with a `language_model_only` flag).
- All satisfy the DeltaNet kernel's constraints: `linear_key_head_dim=128`,
  `linear_value_head_dim=128`, `linear_conv_kernel_dim=4`.
- **Qwen3.7 does not exist** on the Hub. The ladder is 3.5 → 3.6 → 3.8.
- `Qwen/Qwen3.5-4B-GatedDeltaNet` (referenced in `test_preflight.py`) is a fixture,
  not a real repo.

### 5. `publish.py` fails on filesystems without xattrs
`publish.py:141` uses `shutil.copy2`, which copies extended attributes:

```
OSError: [Errno 524] ... os.setxattr   # ENOTSUPP
```

Any S3/FUSE-backed filesystem (including a Kaizen desktop `$HOME`) has no xattr
support, so every publish fails and a run reports `0/8 ok`. With `--out-root /tmp`
the same run reports `8/8 ok`. Use `shutil.copy`, or suppress `copystat`.

### 6. Stale documentation
- `RUN.md` expects "35 passed" and `ENVIRONMENT.md` "31 passed"; the suite is now
  657 tests (633 passed / 24 failed / 2 skipped in the Beta 3 image).
- `RUN.md` step 5 says `backends/native_pytorch.py` is stubbed. It is fully
  implemented (387 lines, no `NotImplementedError`).
- The Beta 3 DLC does **not** ship `transformers`; `native_pytorch.py:40` fails at
  import without it. Pin the version.

## Reproducing

```bash
# Kaizen batch workload — interactive `desktop connect` cannot init a device on trn2.
kz start-workload \
  --command "bash /ustore/fsx/team_shared_rw/scripts/sweep_big2.sh" \
  --image "<beta3-native-dlc>" \
  --instanceType trn2.48xlarge --nodeCount 1 --timeout 28800
```

Inside the workload:

```bash
export TRN_OPT_KERNEL_DIR=/path/to/implementation/src/kernels
python -u run_overnight.py --backend native-pytorch-beta3 \
  --model Qwen/Qwen3.8-27B --family hybrid_attention_causal_lm \
  --kernels-wired --out-root /tmp/art --cycles 1 --max-configs 2
```

Two operational notes: pipe nothing through `tail` (it buffers until exit and makes
a healthy run look hung), and read progress with `kaizen workload exec` rather than
`get-artifact`, which serves a badly stale S3 copy mid-run.

## The scheduler chooses the region, and a wrong region silently voids the run

`kaizen start-workload` has no `--region`: placement is scheduler-side. Band `large2`
was scheduled in **eu-north-1** while every other band landed in **ap-southeast-4**,
and the consequences were invisible:

* a **different FSX filesystem** — so the shared HF cache was cold, and the run would
  have re-downloaded every checkpoint before touching a device
* a **`run.log` no other pod could read**, which is how progress is normally inspected
  (`get-artifact` serves a stale S3 copy mid-run, so reading a sibling pod's log on the
  shared mount is the reliable route)
* the workload reported **SUCCEEDED** having produced nothing

The reason it looked healthy is worth stating plainly, because it will happen again to
anyone writing results to a mount path: **`mkdir -p` on an unmounted path succeeds.**
It creates a perfectly ordinary local directory inside the container, `tee` writes to
it happily, and every byte is deleted with the pod. Nothing anywhere reports an error.
The same trap ate an earlier `gdn_repro` run on a trn2.3xlarge, which does not mount
`FSX_TEAM_SHARED_RW` at all.

So the check cannot be "does the directory exist" — it has to be **"is this path
actually a shared mount"**, read from the mount table, before any work begins:

```bash
FSX=/ustore/fsx/team_shared_rw
EXPECT_REGION=${TRN_OPT_EXPECT_REGION:-ap-southeast-4}
IMDS_T=$(curl -s --max-time 2 -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || true)
REGION=$(curl -s --max-time 2 -H "X-aws-ec2-metadata-token: ${IMDS_T}" \
  http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || true)
[ -z "${REGION}" ] && REGION=${AWS_DEFAULT_REGION:-${AWS_REGION:-unknown}}
echo "PREFLIGHT band=$BAND region=${REGION} expect=${EXPECT_REGION} host=$(hostname)"

# The mount table, not the directory: `mkdir -p` would have hidden this.
if ! grep -q " /ustore/fsx" /proc/mounts 2>/dev/null; then
  echo "PREFLIGHT_FAIL fsx_not_mounted -- results would be deleted with the pod."
  exit 42
fi

CACHE_GB=$(du -s --block-size=1G "$FSX/hf_cache_shared" 2>/dev/null | awk '{print $1+0}')
if [ "$REGION" != "unknown" ] && [ "$REGION" != "$EXPECT_REGION" ]; then
  if [ "${CACHE_GB:-0}" -lt 50 ]; then
    echo "PREFLIGHT_FAIL cold_cache_in_wrong_region cache=${CACHE_GB}GB"
    exit 43
  fi
  echo "PREFLIGHT_WARN wrong region but cache is warm (${CACHE_GB}GB); continuing"
fi
```

Two deliberate choices in there:

**A wrong region is only fatal when the cache is also cold.** The region itself costs
nothing; the cold cache is what burns the slot. Failing on region alone would throw
away usable capacity, and capacity is the scarce thing.

**Preflight prints before any redirect into a file.** If the shared mount is missing,
a log written to that mount is exactly the log nobody can read — so the verdict goes to
stdout, which CloudWatch captures regardless.

Exit codes are distinct (`42` unmounted, `43` cold cache in the wrong region) so a
voided placement is distinguishable from a real failure without reading logs at all.
