"""RoPE sanity/학습 로그 요약: 손실 성분별 처음·에폭 끝 평균 · NaN · init 키 로그. argv: <pretrain .out 로그>..."""
import re, sys

PAT = re.compile(r"Batch (\d+)/(\d+), Loss: ([\d.eE+-]+|nan).*?\| (.*)$")
for path in sys.argv[1:]:
    lines = open(path, errors="replace").read().splitlines()
    epochs, cur = [], None
    for ln in lines:
        m = PAT.search(ln)
        if not m:
            continue
        b = int(m.group(1))
        if b == 0:
            cur = []
            epochs.append(cur)
        comp = dict(kv.split("=") for kv in m.group(4).split())
        comp["Loss"] = m.group(3)
        cur.append({k: float(v) for k, v in comp.items()})
    nan = sum(1 for e in epochs for r in e for v in r.values() if v != v)
    print(f"== {path}")
    for ln in lines:
        if "init loaded" in ln or "rope: dropped" in ln or "[Time]" in ln or "Model args" in ln and "--rope" in ln:
            print("  ", ln.strip()[:200])
    keys = ["Loss", "L_t", "L_tk", "L_pred", "L_mA", "L_mB"]
    mean = lambda rows: {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys if rows and k in rows[0]}
    for i, e in enumerate(epochs, 1):
        print(f"   ep{i}: n_log={len(e)} first5={mean(e[:5])} last5={mean(e[-5:])}")
    print(f"   NaN values: {nan}")
