# src/train.py
import sys, json, time, torch, timm, numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from src.data import get_loaders

MODELS = {"cnn": "resnet18", "vit": "deit_tiny_patch16_224"}
dev = "cuda" if torch.cuda.is_available() else "cpu"

def evaluate(model, loader, max_batches=None):
    model.eval(); P, Y = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches and i >= max_batches: break
            P.append(torch.softmax(model(x.to(dev)), 1).cpu()); Y.append(y.squeeze(1))
    P, Y = torch.cat(P).numpy(), torch.cat(Y).numpy()
    try:
        auroc = roc_auc_score(Y, P, multi_class="ovr")
    except ValueError:
        auroc = float("nan")  # classes manquantes dans le smoke test
    return {"acc": accuracy_score(Y, P.argmax(1)),
            "f1": f1_score(Y, P.argmax(1), average="macro"),
            "auroc": auroc}
    
def main(name, epochs=5, lr=3e-4, subset=None):
    torch.manual_seed(0)
    loaders = get_loaders()
    model = timm.create_model(MODELS[name], pretrained=True, num_classes=9).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best = 0
    for ep in range(epochs):
        model.train(); t0 = time.time()
        for i, (x, y) in enumerate(loaders["train"]):
            if subset and i >= subset: break
            loss = torch.nn.functional.cross_entropy(model(x.to(dev)), y.squeeze(1).long().to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        m = evaluate(model, loaders["val"], max_batches=5 if subset else None)
        print(ep, m, f"{time.time()-t0:.0f}s")
        if m["acc"] > best:
            best = m["acc"]; torch.save(model.state_dict(), f"checkpoints/{name}.pt")
    model.load_state_dict(torch.load(f"checkpoints/{name}.pt"))
    res = evaluate(model, loaders["test"], max_batches=5 if subset else None)
    json.dump(res, open(f"results/{name}_central.json", "w"), indent=2)
    print("TEST", res)

if __name__ == "__main__":
    main(sys.argv[1], subset=int(sys.argv[2]) if len(sys.argv) > 2 else None)