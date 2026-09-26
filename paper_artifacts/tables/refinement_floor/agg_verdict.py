import json,glob,statistics as st,math
R="/proj/home/mrg/bys724/action-agnostic-visual-rl/paper_artifacts"
pos=lambda m: sum(m['r2_per_dim'][:3])/3
arms=["C1","C0","F1","F1proj","F1aug","F1norm","F3"]
def calv(test,a,s):
    names={"F1":"F1_m","F1proj":"F1proj_m","F1aug":"F1aug_m","F1norm":"F1norm_m","F3":"F3_m","C0":"C0_m","C1":"C1_m"}
    d=sorted(glob.glob(f"{R}/calvin_action_probing/*_refine_{test}_{names[a]}_s{s}"))
    if not d and test=="pert" and a!="C1": return None
    import os; return json.load(open(d[-1]+"/gap30/summary.json")) if d and os.path.exists(d[-1]+"/gap30/summary.json") else None
def xfer(a,s):
    d=sorted(glob.glob(f"{R}/libero_action_probing/*_refine_xfer_{a}_m_s{s}"))
    return json.load(open(d[-1]+"/transfer_gap20.json")) if d else None
def ci(v):
    m=st.mean(v); sd=st.stdev(v) if len(v)>1 else float('nan'); h=4.303*sd/math.sqrt(len(v)); return m,sd,m-h,m+h
out=[]
def row(name,vals):
    if len(vals)<3: out.append(f"{name:28s} n={len(vals)} {vals}"); return
    m,sd,lo,hi=ci(vals); out.append(f"{name:28s} {m:+.3f}±{sd:.3f}  CI[{lo:+.3f},{hi:+.3f}]")
out.append("== (A) LIBERO transfer: 6-dir mean pos R2 | same-suite mean | spatial<->goal mean")
for a in arms:
    J=[xfer(a,s) for s in (42,1,2)]; J=[j for j in J if j]
    row(f"{a} transfer6",[j['transfer_pos_r2_mean'] for j in J])
    row(f"{a} insuite",[j['insuite_pos_r2_mean'] for j in J])
    row(f"{a} spatial<->goal",[ (pos(j['matrix']['libero_spatial']['libero_goal'])+pos(j['matrix']['libero_goal']['libero_spatial']))/2 for j in J])
for test in ("label","labelfix"):
    out.append(f"== (B) CALVIN label efficiency [{test}] pos R2 by frac")
    for a in arms:
        J=[calv(test,a,s) for s in (42,1,2)]; J=[j for j in J if j]
        for f in ("1.0","0.2","0.05","0.02"):
            row(f"{a} {f}",[pos(j['label_frac'][f]) for j in J])
out.append("== (C) CALVIN perturbation (C1 + comparison) clean / shadow0.6 / noise0.01 / noise0.04")
for a in arms:
    J=[calv("pert",a,s) for s in (42,1,2)]; J=[j for j in J if j]
    if not J: continue
    row(f"{a} clean",[pos(j) for j in J])
    for k,l in (("shadow","0.6"),("noise","0.01"),("noise","0.04"),("gain","1.3")):
        row(f"{a} {k}{l}",[pos(j['perturb'][k][l]) for j in J])
print("\n".join(out))
