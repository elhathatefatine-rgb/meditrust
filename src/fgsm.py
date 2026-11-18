import sys, os, json, numpy as np, torch, timm
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from src.data import get_datasets
from src.train import MODELS, dev

EPS = [0, 1, 2, 4, 8]  # en unités de 1/255

def fgsm_acc(model, loader, eps):
    model.eval(); ok = n = 0
    for x, y in loader:
        x, y = x.to(dev), y.squeeze(1).long().to(dev)
        x.requires_grad_(True)
        loss = F.cross_entropy(model(x), y)
        grad, = torch.autograd.grad(loss, x)
        xa = (x + 2 * (eps / 255) * grad.sign()).clamp(-1, 1).detach()
        with torch.no_grad():
            ok += (model(xa).argmax(1) == y).sum().item(); n += len(y)
    return ok / n

def main(ckpt, name="vit", n=2000):
    tag = os.path.basename(ckpt).replace(".pt", "")
    ds = get_datasets()["test"]
    idx = np.random.default_rng(0).choice(len(ds), n, replace=False).tolist()
    loader = DataLoader(Subset(ds, idx), batch_size=32)
    model = timm.create_model(MODELS[name], pretrained=False, num_classes=9).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    res = {}
    for e in EPS:
        res[f"{e}/255"] = fgsm_acc(model, loader, e); print(f"eps={e}/255  acc={res[f'{e}/255']:.4f}")
    json.dump(res, open(f"results/fgsm_{tag}.json", "w"), indent=2)
    plt.plot(EPS, list(res.values()), "o-"); plt.xlabel("ε (×1/255)"); plt.ylabel("Accuracy")
    plt.title(f"FGSM sur {tag}"); plt.grid(True); plt.savefig(f"results/fgsm_{tag}.png", dpi=150)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "vit")