import json, copy, argparse, numpy as np, torch, timm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from src.data import get_datasets
from src.train import MODELS, dev

def swap(x):  # hôpital B : coloration différente (canaux R et B échangés)
    return x[:, [2, 1, 0]]

@torch.no_grad()
def acc(model, loader, shift=False):
    model.eval(); ok = n = 0
    for x, y in loader:
        x = x.to(dev); x = swap(x) if shift else x
        ok += (model(x).argmax(1).cpu() == y.squeeze(1).long()).sum().item(); n += len(y)
    return ok / n

def fisher_diag(model, loader):  # loader en batch de 1 : gradients par échantillon
    model.eval(); n = 0
    fisher = {k: torch.zeros_like(p) for k, p in model.named_parameters()}
    for x, y in loader:
        model.zero_grad()
        F.cross_entropy(model(x.to(dev)), y.squeeze(1).long().to(dev)).backward()
        for k, p in model.named_parameters():
            if p.grad is not None: fisher[k] += p.grad.detach() ** 2
        n += 1
    mean = sum(v.sum() for v in fisher.values()) / sum(v.numel() for v in fisher.values()) / n
    return {k: v / n / mean for k, v in fisher.items()}  # normalisé : moyenne = 1

def finetune(model, loader, fisher, star, lam, epochs, lr):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    for _ in range(epochs):
        for x, y in loader:
            x, y = swap(x.to(dev)), y.squeeze(1).long().to(dev)
            loss = F.cross_entropy(model(x), y)
            if lam > 0:
                loss = loss + lam / 2 * sum((fisher[k] * (p - star[k]) ** 2).sum()
                                            for k, p in model.named_parameters())
            opt.zero_grad(); loss.backward(); opt.step()

def main(a):
    torch.manual_seed(0); rng = np.random.default_rng(0)
    ds = get_datasets()
    idx_b = rng.choice(len(ds["train"]), a.n_b, replace=False).tolist()
    idx_f = rng.choice(len(ds["train"]), a.n_fisher, replace=False).tolist()
    idx_t = rng.choice(len(ds["test"]), 2000, replace=False).tolist()
    ld_b = DataLoader(Subset(ds["train"], idx_b), batch_size=64, shuffle=True)
    ld_f = DataLoader(Subset(ds["train"], idx_f), batch_size=1)
    ld_t = DataLoader(Subset(ds["test"], idx_t), batch_size=64)
    base = timm.create_model(MODELS["vit"], pretrained=False, num_classes=9).to(dev)
    base.load_state_dict(torch.load(a.ckpt, map_location=dev))
    fisher = fisher_diag(base, ld_f)
    star = {k: p.detach().clone() for k, p in base.named_parameters()}
    res = {"avant": {"acc_A": acc(base, ld_t), "acc_B": acc(base, ld_t, True)}}
    print("avant", res["avant"])
    for lam in a.lams:
        m = copy.deepcopy(base)
        finetune(m, ld_b, fisher, star, lam, a.epochs, a.lr)
        res[f"lambda={lam:g}"] = {"acc_A": acc(m, ld_t), "acc_B": acc(m, ld_t, True)}
        print(f"lambda={lam:g}", res[f"lambda={lam:g}"])
    json.dump(res, open("results/continual.json", "w"), indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--lams", type=float, nargs="+", default=[0, 1, 10, 100, 1000])
    p.add_argument("--n_b", type=int, default=10000)
    p.add_argument("--n_fisher", type=int, default=500)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    main(p.parse_args())