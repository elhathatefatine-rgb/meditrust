import sys, os, json, numpy as np, torch, timm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from src.data import get_datasets
from src.train import MODELS, dev

EPS = [1, 2, 4]  # en unités de 1/255

def pgd_acc(model, loader, eps, steps=10):
    e = 2 * eps / 255; alpha = 2.5 * e / steps
    model.eval(); ok = n = 0
    for x, y in loader:
        x, y = x.to(dev), y.squeeze(1).long().to(dev)
        d = torch.empty_like(x).uniform_(-e, e)
        for _ in range(steps):
            d.requires_grad_(True)
            g, = torch.autograd.grad(F.cross_entropy(model((x + d).clamp(-1, 1)), y), d)
            d = (d.detach() + alpha * g.sign()).clamp(-e, e)
        with torch.no_grad():
            ok += (model((x + d).clamp(-1, 1)).argmax(1) == y).sum().item(); n += len(y)
    return ok / n

def main(ckpt, name="vit", n=1000):
    tag = os.path.basename(ckpt).replace(".pt", "")
    ds = get_datasets()["test"]
    idx = np.random.default_rng(0).choice(len(ds), n, replace=False).tolist()
    loader = DataLoader(Subset(ds, idx), batch_size=32)
    model = timm.create_model(MODELS[name], pretrained=False, num_classes=9).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    res = {}
    for e in EPS:
        res[f"{e}/255"] = pgd_acc(model, loader, e)
        print(f"PGD-10 eps={e}/255  acc={res[f'{e}/255']:.4f}")
    json.dump(res, open(f"results/pgd_{tag}.json", "w"), indent=2)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "vit")