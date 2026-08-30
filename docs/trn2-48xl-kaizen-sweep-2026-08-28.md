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

## Tensor parallelism only works at power-of-two world sizes

`#140` widened the TP search from powers of two to **every divisor** of the query-head
count. The reasoning was sound on paper: a 24-head model capped at tp=8 leaves 40 of a
64-core box idle, and tp=12 and tp=24 shard 24 heads perfectly evenly. The hardware
disagrees.

Measured directly, one `init_process_group` + one `all_reduce` per world size, no model
loaded at all:

| world size | result |
|-----------:|:-------|
| 2 | collective forms, `all_reduce` correct |
| 3 | `RuntimeError: Failed to execute the device barrier 2` |
| 4 | forms, correct |
| 5 | barrier failure |
| 6 | barrier failure |
| 12 | barrier failure |
| 16 | forms, correct |
| 24 | barrier failure |
| 32 | forms, correct |
| 64 | forms, correct |

Every non-power-of-two fails. (world=8 also failed in that sequence, immediately after
the world=6 failure — a crashed run leaves the runtime unable to initialise the next
one, so the container needs restarting between TP attempts. tp=8 is the working baseline
for Qwen3.8-27B, so 8 is fine on a clean runtime.)

### What it cost to learn this the other way

The Qwen3.8-27B sweep proposed the full divisor set and paid a **55 GB checkpoint load
per candidate** to discover the same thing:

```
tp_degree=3   -> collective/TP initialisation failed (init_process_group)
tp_degree=6   -> collective/TP initialisation failed
tp_degree=12  -> collective/TP initialisation failed
tp_degree=24  -> collective/TP initialisation failed
```

Four config slots, no information gained about the model.

### The corrected rule

TP candidates are the **intersection**: powers of two that divide the head count. Both
filters are load-bearing — drop the divisor check and tp=16 comes back for a 24-head
model (rejected by the worker as `invalid_tp`); drop the ladder and tp=12 comes back
(rejected by the runtime).

| model | heads | reachable TP | note |
|:--|--:|:--|:--|
| Qwen3.8-27B | 24 | 1, 2, 4, 8 | **tp=24 unreachable** |
| Qwen3.5-35B-A3B | 16 | 1, 2, 4, 8, 16 | |
| Qwen3.5-122B-A10B | 32 | 1, 2, 4, 8, 16, 32 | |
| DeepSeek-V4-Flash | 64 | 1 … 64 | |
| MiniMax-M2 | 48 | 1, 2, 4, 8, 16 | **tp=48 unreachable** |

### Two consequences worth stating plainly

**Idle cores on a low-head-count model are not a config bug.** Qwen3.5-0.8B has 8 query
heads and won at tp=4; 60 of 64 cores are capacity that model cannot address by
sharding. The way to use them is more concurrent **replicas**, which is what the
box-throughput figure measures — 9,104 tok/s from 12 replicas at tp=4, against 1,143
tok/s for one.

**MiniMax-M2 does not fit by tensor parallelism.** The capability gate previously
admitted it on the strength of tp=48 giving 9.6 GB/rank. At the tp=16 it can actually
form, 460 GB is 29 GB/rank against a 14 GB/rank budget, so the honest verdict is
`TOO_LARGE`. It needs expert parallelism or a second node — and an over-predicting gate
is worse than a rejecting one, because it sends a band off to spend hours rediscovering
the limit.

## The host-DRAM model under-predicts, measured on a 470 GB load

Band `huge2` was killed loading Qwen3-235B-A22B-Instruct-2507. Nothing appears in its
`run.log` past `establishing baseline` because the process was killed by the kernel, not
by Python — the evidence is in the memory monitor:

```
[mem 10:35:01] 29 GB avail
[mem 10:36:01] 18 GB avail
[mem 10:36:31] 12 GB avail
[mem 10:37:01]  6 GB avail      <- of 2147 GB total
```

The capability gate had cleared this configuration:

```
Qwen/Qwen3-235B-A22B-Instruct-2507   470.2 GB, 64 heads, tp=64
  conc=1: ok=True RUNNABLE   470 GB at tp=64 = 7.3 GB/rank, within the 14 GB/rank budget
  conc=2: ok=True RUNNABLE
  conc=3: ok=True RUNNABLE   <- what was launched
```

HBM was never the problem: 7.3 GB/rank against a 14.4 GB budget is comfortable. The
host is what ran out. The gate's host model reckons `concurrency` full copies plus a
shard on every other rank, which for this model at conc=3 is

    3 x 470 GB  +  61 x 7.3 GB  ~=  1858 GB     against 2147 GB available

so it passed with ~289 GB of headroom, and the real peak still exceeded 2141 GB. The
model is therefore **low by at least ~15%** on a load of this size. Candidate causes,
none yet isolated: page cache from streaming 470 GB of safetensors, a materialized copy
alongside the mmap, or ranks holding a full state dict past the point the stagger model
assumes they have dropped to shard size.

One data point does not justify fitting a correction factor, so the model is unchanged
and the number is recorded here instead. What follows from it operationally:

**Treat `RUNNABLE` at high concurrency as unproven for anything over ~400 GB.** The
relaunch uses `TRN_OPT_LOAD_CONCURRENCY=1`, where the same model is predicted at
470 + 448 = ~918 GB — less than half of host DRAM, so the ~15% error cannot reach the
ceiling.

**The failure mode is silent.** A kernel OOM kill leaves no Python traceback and the
workload simply reports FAILED, three hours after the last log line. The `mem.log`
sidecar the band template writes is the only reason this was diagnosable at all, which
is a good argument for keeping it.

MiniMax-Text-01 is the case where the model already says no: 915 GB is `HOST_LIMITED`
at conc=2 (2715 GB needed) and only `TIGHT` at conc=1 (14.3 GB/rank against a 14.4
ceiling). Given the under-prediction above, its conc=1 verdict deserves the same
scepticism.

## Two cheap causes that cost whole models

Band `large2` ran three P0/P1 targets and published none. Neither failure was about
optimization:

**Kimi-Linear-48B-A3B-Instruct — a missing 40 KB package.** 34 seconds from start to
crash:

```
[kimi-linear-48b-a3b-instruct] CRASHED: FAIL_NO_BASELINE: ... baseline produced no
throughput (metric=0.0) ... WORKER: rc=1: [rank7]: ImportError: This modeling file
requires the following packages that were not found in your environment: einops.
```

`einops` is a hard requirement of several remote modeling files. The orchestrator handled
it correctly — no baseline means no incumbent to optimize, so it refused to continue —
but the whole model was lost to a `pip install`. `einops`, `sentencepiece` and `tiktoken`
are now installed by the band template.

**DeepSeek-V4-Flash — runtime contamination from the previous model.** 35 seconds, and
the reason it failed is *when* it ran:

```
09:26 -> 11:34  Qwen3.5-122B-A10B   grader: unverified (claimed=6 remeasured=0
                                    drift=100.0% equivalence=FAIL)
                                    re-measure failed: collective/TP initialisation failed
11:34 -> 11:35  DeepSeek-V4-Flash   FAIL_NO_BASELINE: collective/TP initialisation failed
```

DeepSeek-V4-Flash has 64 query heads, and world=64 initialises cleanly on a fresh
container (measured). What it inherited was a runtime left broken by the 122B's failed
collective in the same container — the same effect that makes a TP ladder unreliable
after its first failure.

So **one model per workload for anything that shards wide.** Sequencing large models in a
single container means the first collective failure silently poisons every model after it,
and the symptom is indistinguishable from the later model being broken. `large2` looked
like three unrelated failures and was really one missing package plus one carried-over
runtime.

Worth noting what worked correctly here: the 122B claimed 6 tok/s, the trusted grader
re-measured it, the re-measure failed, and the result was marked `unverified` and **not
published**. A claimed number that cannot be reproduced never reaches the board.

## Qwen3-30B-A3B fails at baseline with NRT_RESOURCE, on both load paths

Qwen3-30B-A3B has now failed to establish a baseline five times -- `mid3`, `sorval`,
`sorval2`, `moe30` -- with the same error, and the last two ran with shard-on-read ON:

```
[measure] worker produced no result (rc=1): [rank0]: NRT EXECUTION FAILED:
lazy::AllocBind: NRT_RESOURCE; no pending ops on stream to wait for (cannot defer),
Failed to allocate resource
-> CRASHED: FAIL_NO_BASELINE (metric=0.0)
```

What this rules out, and why it matters for where to spend effort:

* **Not a loading problem.** shard-on-read streams each rank's expert slice off disk
  and never materialises the full model, so if the failure were host-DRAM at load, it
  would have changed. It did not -- byte-identical error with the flag on and off. The
  crash is `AllocBind` during graph *execution*, after the weights are in place.

* **Not the host-DRAM ceiling.** `AllocBind ... Failed to allocate resource` with
  "no pending ops on stream to wait for (cannot defer)" is the Neuron runtime unable to
  allocate a *device* resource during execution setup, having found nothing to evict.
  This is HBM or a runtime object (DMA rings, etc.), not the 2 TB host wall the 235B
  hit.

* **The weight arithmetic says it should fit.** `_fit_baseline_tp` picks the smallest
  tp with `weight_gb/tp < 10`; for 30B-A3B (~60 GB bf16) that is tp=8 -> 7.5 GB/rank,
  and at tp=8 the worker shards both experts (EP) and attention (TP). 7.5 GB of weights
  on a 24 GB core should leave ample room, so the resource being exhausted is not
  simply the weights -- it points at activation/scratch for the 128-expert routing at
  the baseline sequence length, or a per-rank runtime-object limit that the MoE graph
  trips.

So this is a distinct, undiagnosed blocker sitting *in front of* every large-MoE win on
this box: the model cannot be scored because its baseline will not run, before any
optimisation is attempted. It is separate from both the host-DRAM loading problem
(which shard-on-read addresses) and the `NCC_IINAR001` compiler bug (#134, the
GatedDeltaNet family). Worth its own investigation -- candidate next steps: capture
`neuron-ls`/runtime resource state at the failure, try the baseline at tp=16 (halve
per-rank pressure), and shorten the baseline sequence to isolate activation memory.

## 235B: shard-on-read solved the LOAD, and exposed a second host-DRAM wall at COMPILE

Ran Qwen3-235B-A22B with `TRN_OPT_SHARD_ON_READ=1` (band `huge4`). The result is the
clearest evidence yet for shard-on-read, and it uncovered the next distinct blocker.

**Shard-on-read works, measured.** After the 470 GB checkpoint finished loading, host
DRAM sat at **1,586 GB free of 2,147**. The full-load path (`huge2`, `huge3`) fell to
**6-14 GB free** at the same phase and was OOM-killed. So the model loaded on one node
without ever materialising the full weights per rank -- exactly the design, on the
biggest model in the plan, on real hardware.

**Then it OOM-killed during the baseline COMPILE.** Watched live, host DRAM fell
steadily once the first forward began:

```
15:54  1586 GB free   (load complete -- shard-on-read holding)
17:36   393 GB free
17:40   222 GB free
17:41   190 GB free    (exec on the pod starts hanging -- memory pressure)
18:01   FAILED
```

The cause is not loading and not HBM: it is **64 parallel `neuronx-cc` processes** --
one per rank at tp=64 -- each compiling a slice of a 235B / 128-expert graph, each
consuming multiple GB of HOST RAM at once. `load_stagger` bounds the load window but
not the compile, and the compile cannot be staggered the same way: the first forward is
a tensor-parallel collective, so holding ranks back to serialise their compiles would
deadlock the all-reduce.

So there are now three separable walls in front of the large models, and this session
moved the first:

1. **Load host-DRAM** -- SOLVED by shard-on-read (proven above).
2. **Compile host-DRAM** -- NEW. 64 concurrent `neuronx-cc` on a huge graph. Candidate
   fixes: cap per-process compiler host-RAM via `NEURON_CC_FLAGS`; a shared on-FSX
   compile cache so re-compiles are free (does not help the first simultaneous 64,
   though); or a smaller baseline (fewer layers compiled at once) to get a first
   verified number, then scale.
3. **Baseline-forward HBM / NRT_RESOURCE** -- the mid-size MoE wall (30B, Kimi-Linear).

Worth stating plainly: shard-on-read did its job. 235B is the first time the full model
ever loaded on a single 48xl. The compile wall is a different problem, and it is the one
to solve next for the very largest models -- while the mid-size MoEs are gated on wall 3
instead.
