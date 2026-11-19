import sys, os, json, numpy as np, torch, timm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from medmnist import INFO
from src.data import get_datasets
from src.train import MODELS, dev

@torch.no_grad()
def predict_all(model, loader):
    model.eval(); P, Y = [], []
    for x, y in loader:
        P.append(model(x.to(dev)).softmax(1).cpu()); Y.append(y.squeeze(1))
    return torch.cat(P).numpy(), torch.cat(Y).long().numpy()

def main(ckpt):
    tag = os.path.basename(ckpt).replace(".pt", "")
    names = [INFO["pathmnist"]["label"][str(i)] for i in range(9)]
    model = timm.create_model(MODELS["vit"], pretrained=False, num_classes=9).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    P, Y = predict_all(model, DataLoader(get_datasets()["test"], batch_size=128))
    pred, conf = P.argmax(1), P.max(1)
    cm = confusion_matrix(Y, pred, labels=range(9))
    k, nrm = names.index("colorectal adenocarcinoma epithelium"), names.index("normal colon mucosa")
    m = Y == k; miss = m & (pred == nrm)
    out = {"acc": float((pred == Y).mean()), "classes": names, "confusion": cm.tolist(),
           "recall_par_classe": {n: float(cm[i, i] / cm[i].sum()) for i, n in enumerate(names)},
           "cancer_total": int(m.sum()), "cancer_predit_normal": int(miss.sum()),
           "cancer_predit_normal_conf_sup_0.9": int((miss & (conf >= 0.9)).sum())}
    print(json.dumps({k_: v for k_, v in out.items() if k_ != "confusion"}, indent=2, ensure_ascii=False))
    json.dump(out, open(f"results/confusion_{tag}.json", "w"), indent=2, ensure_ascii=False)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7)); ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(9)); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(9)); ax.set_yticklabels(names, fontsize=7)
    for i in range(9):
        for j in range(9):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Prédit"); ax.set_ylabel("Vrai"); plt.tight_layout()
    plt.savefig(f"results/confusion_{tag}.png", dpi=150)

if __name__ == "__main__":
    main(sys.argv[1])