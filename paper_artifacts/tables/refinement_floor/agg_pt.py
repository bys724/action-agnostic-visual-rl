import json,glob,os,statistics as st,math
R="/proj/home/mrg/bys724/action-agnostic-visual-rl/paper_artifacts"
pos=lambda m: sum(m['r2_per_dim'][:3])/3
def ld(pat,f):
    d=sorted(glob.glob(pat)); p=d[-1]+f if d else None
    return json.load(open(p)) if p and os.path.exists(p) else None
def ci(v):
    if len(v)<2: return f"n={len(v)} {v}"
    m=st.mean(v); sd=st.stdev(v); h=4.303*sd/math.sqrt(len(v)); return f"{m:+.3f}±{sd:.3f} CI[{m-h:+.3f},{m+h:+.3f}] (n={len(v)})"
for a in ("PtC1M","PtRaw"):
    print("=====",a)
    X=[ld(f"{R}/libero_action_probing/*_refine_xfer_{a}_s{s}","/transfer_gap20.json") for s in (42,1,2)]; X=[x for x in X if x]
    print(" xfer transfer6 ", ci([x['transfer_pos_r2_mean'] for x in X]))
    print(" xfer insuite   ", ci([x['insuite_pos_r2_mean'] for x in X]))
    print(" xfer spat<->goal", ci([(pos(x['matrix']['libero_spatial']['libero_goal'])+pos(x['matrix']['libero_goal']['libero_spatial']))/2 for x in X]))
    P=[ld(f"{R}/calvin_action_probing/*_refine_pert_{a}_s{s}","/gap30/summary.json") for s in (42,1,2)]; P=[p for p in P if p]
    print(" pert clean     ", ci([pos(p) for p in P]))
    for k,l in (("shadow","0.2"),("shadow","0.4"),("shadow","0.6"),("noise","0.01"),("noise","0.04"),("gain","1.1"),("gain","1.3"),("ramp","0.3")):
        print(f" pert {k}{l:5s}", ci([pos(p['perturb'][k][l]) for p in P]))
    L=[ld(f"{R}/calvin_action_probing/*_refine_label_{a}_s{s}","/gap30/summary.json") for s in (42,1,2)]; L=[x for x in L if x]
    for f in ("1.0","0.2","0.05","0.02"):
        print(f" label {f:5s}   ", ci([pos(x['label_frac'][f]) for x in L]))
