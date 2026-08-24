"""On-device driver for the self-improvement loop (Pillar 2/3).

Runs run_selfimprove for one or more ops with the real Bedrock Opus-5 author,
writes per-op trajectory JSON, and prints a compact skill-curve table.
"""
import argparse, json, sys, time
from pathlib import Path

import nki_selfimprove as SI


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default="softmax")
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--out", default="/home/ubuntu/selfimprove_run")
    ap.add_argument("--stop-k", type=int, default=3)
    ap.add_argument("--model", default="global.anthropic.claude-opus-5")
    ap.add_argument("--region", default="ap-southeast-4")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]

    def log(msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    results = {}
    for op in ops:
        log(f"==== self-improve {op} x{args.iters} ====")
        res = SI.run_selfimprove(
            op, iters=args.iters,
            out_dir=out / op,
            provider="bedrock", model_id=args.model, region=args.region,
            no_improve_stop_k=args.stop_k, log=log)
        results[op] = res
        (out / f"{op}.trajectory.json").write_text(json.dumps(res, indent=2))
        log(f"---- {op} summary: {json.dumps(res['summary'])}")

    (out / "all_results.json").write_text(json.dumps(results, indent=2))
    # Compact skill-curve table.
    print("\n================ SKILL CURVES ================", flush=True)
    for op, res in results.items():
        print(f"\n## {op}  ({res['shape_class']})  stop={res['stop_reason']}")
        print(f"{'iter':>4} {'status':>13} {'cmpl':>5} {'corr':>5} "
              f"{'speedup':>8} {'best':>8} {'class':>20} {'lesson':>6} {'chg':>4} {'dt_s':>6}")
        for r in res["trajectory"]:
            sp = f"{r['speedup']:.3f}" if r.get("speedup") is not None else "  -  "
            bs = f"{r['best_speedup_so_far']:.3f}" if r.get("best_speedup_so_far") is not None else "  -  "
            print(f"{r.get('iteration','?'):>4} {str(r.get('status'))[:13]:>13} "
                  f"{str(r.get('compiled')):>5} {str(r.get('correct')):>5} "
                  f"{sp:>8} {bs:>8} {str(r.get('outcome_class'))[:20]:>20} "
                  f"{str(r.get('lesson_injected')):>6} "
                  f"{str(r.get('approach_changed_from_prev')):>4} {r.get('dt_s','?'):>6}")
        s = res["summary"]
        print(f"  -> first_try_compiled={s['first_try_compiled']} "
              f"rounds_to_correct={s['rounds_to_correct']} "
              f"iter1_speedup={s['iter1_speedup']} best={s['best_speedup']} "
              f"improved_over_iter1={s['improved_over_iter1']} "
              f"approach_changes={s['approach_changes']}")


if __name__ == "__main__":
    main()
