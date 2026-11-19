import os, torch, medmnist
from app import NETS, tf, LABELS

d = medmnist.PathMNIST(split="test", size=64, root="data")
fp32 = NETS["Standard"][0]
os.makedirs("examples", exist_ok=True)
N = 1500
errors, lowconf = [], []
for s in range(0, N, 64):
    idx = range(s, min(s + 64, N))
    x = torch.stack([tf(d[i][0].convert("RGB")) for i in idx])
    p = fp32(x).softmax(1)
    for k, i in enumerate(idx):
        y = int(d[i][1][0]); c, pred = p[k].max(0); pred = int(pred); c = float(c)
        if pred != y: errors.append((i, y, pred, c))
        elif c < 0.9: lowconf.append((i, y, pred, c))
errors.sort(key=lambda t: -t[3])
print(f"{len(errors)} erreurs sur {N} images, dont {sum(e[3] < 0.9 for e in errors)} sous le seuil 0.9")
print(f"{len(lowconf)} prédictions justes mais sous le seuil 0.9")
for tag, lst in [("err", errors[:4]), ("low", lowconf[:3])]:
    for i, y, pred, c in lst:
        d[i][0].save(f"examples/{tag}_{i}.png")
        print(f"{tag}_{i}.png : vrai={LABELS[y]} | pred={LABELS[pred]} ({c:.2f})")