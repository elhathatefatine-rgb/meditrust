import json, glob, numpy as np

CFG = {"Centralisé (K=1)": "fed_k1_a0.5",
       "Fédéré IID (α=100)": "fed_k5_a100.0",
       "Fédéré non-IID (α=0.5)": "fed_k5_a0.5"}
for name, base in CFG.items():
    files = [f"results/{base}.json"] + sorted(glob.glob(f"results/{base}_s[0-9]*.json"))
    T = [json.load(open(f))["test"] for f in files]
    cols = []
    for m in ["acc", "f1", "auroc"]:
        v = np.array([t[m] for t in T])
        cols.append(f"{m}={v.mean():.4f}±{(v.std(ddof=1) if len(v) > 1 else 0):.4f}")
    print(f"{name:24s} n={len(T)}  " + "  ".join(cols))
    