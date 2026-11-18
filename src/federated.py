import json, copy, argparse
import numpy as np, torch, timm
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from src.data import get_datasets
from src.train import evaluate, MODELS, dev

def dirichlet_split(labels, k, alpha, rng):
    idx = [[] for _ in range(k)]
    for c in np.unique(labels):
        ids = np.where(labels == c)[0]; rng.shuffle(ids)
        p = rng.dirichlet([alpha] * k)
        cuts = (np.cumsum(p) * len(ids)).astype(int)[:-1]
        for i, part in enumerate(np.split(ids, cuts)):
            idx[i] += part.tolist()
    return idx

def local_train(model, loader, epochs, lr):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    for _ in range(epochs):
        for x, y in loader:
            loss = F.cross_entropy(model(x.to(dev)), y.squeeze(1).long().to(dev))
            opt.zero_grad(); loss.backward(); opt.step()

def fedavg(states, sizes):
    tot = sum(sizes); avg = {}
    for k in states[0]:
        if states[0][k].is_floating_point():
            avg[k] = sum(s[k].float() * (n / tot) for s, n in zip(states, sizes)).to(states[0][k].dtype)
        else:
            avg[k] = states[0][k]
    return avg

def main(a):
    torch.manual_seed(0); rng = np.random.default_rng(0)
    ds = get_datasets()
    labels = ds["train"].labels.squeeze()
    pool = rng.choice(len(labels), a.n_total, replace=False)
    parts = [pool[np.array(p, dtype=int)] for p in dirichlet_split(labels[pool], a.clients, a.alpha, rng)]
    parts = [p for p in parts if len(p) > 0]
    counts = [np.bincount(labels[p], minlength=9).tolist() for p in parts]
    for i, c in enumerate(counts): print(f"client {i}: n={sum(c)} classes={c}")

    loaders = [DataLoader(Subset(ds["train"], p.tolist()), batch_size=a.bs, shuffle=True) for p in parts]
    val = DataLoader(ds["val"], batch_size=a.bs)
    test = DataLoader(ds["test"], batch_size=a.bs)
    glob = timm.create_model(MODELS[a.model], pretrained=True, num_classes=9).to(dev)

    log = []
    for r in range(a.rounds):
        states, sizes = [], []
        for p, ld in zip(parts, loaders):
            local = copy.deepcopy(glob)
            local_train(local, ld, a.local_epochs, a.lr)
            states.append({k: v.detach().cpu().clone() for k, v in local.state_dict().items()})
            sizes.append(len(p)); del local
        glob.load_state_dict(fedavg(states, sizes))
        m = evaluate(glob, val, max_batches=40)
        log.append({"round": r, **m}); print(r, m)

    res = evaluate(glob, test)
    print("TEST", res)
    tag = f"k{a.clients}_a{a.alpha}"
    torch.save(glob.state_dict(), f"checkpoints/fed_{tag}.pt")
    json.dump({"config": vars(a), "client_class_counts": counts, "rounds": log, "test": res},
              open(f"results/fed_{tag}.json", "w"), indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vit")
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--n_total", type=int, default=20000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bs", type=int, default=64)
    main(p.parse_args())