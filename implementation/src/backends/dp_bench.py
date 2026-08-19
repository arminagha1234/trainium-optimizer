"""DP fill-the-box benchmark: launch `dp` INDEPENDENT tp-sharded replicas on
disjoint NeuronCore slices, concurrently, and sum real per-replica prefill
throughput. Reuses the worker unchanged as the per-replica runner."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
WORKER=str(Path(__file__).resolve().parent / "neuron_worker.py")  # the LIVE worker
ap=argparse.ArgumentParser()
ap.add_argument("--model",required=True); ap.add_argument("--tp",type=int,default=1)
ap.add_argument("--dp",type=int,default=2); ap.add_argument("--batch",type=int,default=1)
ap.add_argument("--input-len",type=int,default=512); ap.add_argument("--base-core",type=int,default=0)
ap.add_argument("--compile",type=int,default=0)   # pass the winner's compile mode
ap.add_argument("--attn",default="eager")
ap.add_argument("--out",required=True)
a=ap.parse_args()
src=Path(a.model).name.replace(".","_")
outdir=Path(f"/tmp/dpbench_{src}"); outdir.mkdir(exist_ok=True)
procs=[]; t0=time.time()
for r in range(a.dp):
    lo=a.base_core + r*a.tp; hi=lo + a.tp - 1
    env={**os.environ,"NEURON_RT_VISIBLE_CORES":f"{lo}-{hi}",
         "HF_HUB_OFFLINE":"1","HF_HUB_DISABLE_PROGRESS_BARS":"1","TOKENIZERS_PARALLELISM":"false"}
    outf=str(outdir/f"r{r}.json")
    try: os.remove(outf)
    except OSError: pass
    # SATURATION GUARD: `nice` every replica so the OS/sshd always keeps CPU
    # (the un-niced 60-replica burst previously starved sshd). Stagger launches
    # so N model loads don't hit RAM/CPU simultaneously.
    cmd=["nice","-n","15","torchrun","--nproc_per_node",str(a.tp),"--rdzv_backend","c10d",
         "--rdzv_endpoint",f"localhost:{29700+r}",WORKER,"--model",a.model,"--tp",str(a.tp),
         "--dtype","bf16","--attn",a.attn,"--compile",str(a.compile),"--input-len",str(a.input_len),
         "--batch",str(a.batch),"--iters","10","--out",outf]
    procs.append((r,outf,subprocess.Popen(cmd,env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)))
    time.sleep(0.5)   # stagger to avoid a thundering-herd of model loads
for r,outf,p in procs: p.wait()
toks=[]; hbm=[]; okc=0
for r,outf,p in procs:
    try:
        d=json.load(open(outf))
        if d.get("ok"): okc+=1; toks.append(d["tok_s"]); hbm.append(d.get("hbm_peak_gb",0))
    except Exception: pass
per=(sum(toks)/len(toks)) if toks else 0.0
agg=sum(toks)
res={"model":a.model,"tp":a.tp,"dp":a.dp,"replicas_ok":okc,"batch":a.batch,
     "input_len":a.input_len,"per_replica_mean_tok_s":round(per,1),
     "aggregate_tok_s":round(agg,1),"wall_s":round(time.time()-t0,1),
     "per_replica":[round(t,1) for t in toks],"hbm_peak_gb_max":round(max(hbm),2) if hbm else 0}
json.dump(res,open(a.out,"w"),indent=2)
print("DPBENCH", json.dumps(res))
