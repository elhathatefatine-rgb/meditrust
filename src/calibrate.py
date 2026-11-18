import sys, os, json, torch, timm
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.data import get_datasets
from src.train import MODELS, dev

@torch.no_grad()
def get_logits(model, loader):
    model.eval(); L, Y = [], []
    for x, y in loader:
        L.append(model(x.to(dev)).cpu()); Y.append(y.squeeze(1))
    return torch.cat(L), torch.cat(Y).long()

def ece_score(probs, y, n_bins=15):
    conf, pred = probs.max(1); acc = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean().item() * abs(acc[m].mean().item() - conf[m].mean().item())
    return e

def fit_temperature(logits, y):
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(logits / logT.exp(), y); loss.backward(); return loss
    opt.step(closure)
    return logT.exp().item()

def rel_points(probs, y, n_bins=10):
    conf, pred = probs.max(1); acc = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1); xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            xs.append(conf[m].mean().item()); ys.append(acc[m].mean().item())
    return xs, ys

def risk_coverage(probs, y):
    conf, pred = probs.max(1); order = conf.argsort(descending=True)
    n = len(y); err = (pred[order] != y[order]).float().cumsum(0) / torch.arange(1, n + 1)
    return (torch.arange(1, n + 1) / n).numpy(), err.numpy()

def main(ckpt, name="vit"):
    tag = os.path.basename(ckpt).replace(".pt", "")
    ds = get_datasets()
    val = DataLoader(ds["val"], batch_size=128); test = DataLoader(ds["test"], batch_size=128)
    model = timm.create_model(MODELS[name], pretrained=False, num_classes=9).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    lv, yv = get_logits(model, val); lt, yt = get_logits(model, test)
    T = fit_temperature(lv, yv)
    res = {"T": T}
    for s, (l, y) in {"val": (lv, yv), "test": (lt, yt)}.items():
        for when, t in [("before", 1.0), ("after", T)]:
            p = F.softmax(l / t, 1)
            res[f"ece_{s}_{when}"] = ece_score(p, y)
            res[f"nll_{s}_{when}"] = F.nll_loss(p.log(), y).item()
    print(json.dumps(res, indent=2))
    json.dump(res, open(f"results/calibration_{tag}.json", "w"), indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for a, (title, t) in zip(ax[:2], [("Avant", 1.0), (f"Après (T={T:.2f})", T)]):
        xs, ys = rel_points(F.softmax(lt / t, 1), yt)
        a.plot([0, 1], [0, 1], "k--"); a.plot(xs, ys, "o-")
        a.set_title(f"Fiabilité test, {title}"); a.set_xlabel("Confiance"); a.set_ylabel("Accuracy")
    for lbl, t in [("avant", 1.0), ("après", T)]:
        c, e = risk_coverage(F.softmax(lt / t, 1), yt); ax[2].plot(c, e, label=lbl)
    ax[2].set_title("Risque-couverture (test)"); ax[2].set_xlabel("Couverture"); ax[2].set_ylabel("Erreur"); ax[2].legend()
    plt.tight_layout(); plt.savefig(f"results/calibration_{tag}.png", dpi=150)

if __name__ == "__main__":
    main(sys.argv[1])