import sys, os, numpy as np, torch, timm
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from medmnist import INFO
from src.data import get_datasets
from src.train import MODELS, dev

def load(ckpt, name="vit"):
    m = timm.create_model(MODELS[name], pretrained=False, num_classes=9)
    m.load_state_dict(torch.load(ckpt, map_location=dev)); m.to(dev).eval()
    for b in m.blocks: b.attn.fused_attn = False  # expose les poids d'attention
    return m

@torch.no_grad()
def rollout(model, x):
    maps = []
    hooks = [b.attn.attn_drop.register_forward_hook(lambda m, i, o: maps.append(o.detach()))
             for b in model.blocks]
    logits = model(x)
    for h in hooks: h.remove()
    n = maps[0].shape[-1]; eye = torch.eye(n, device=x.device)
    R = eye.expand(x.shape[0], n, n)
    for a in maps:
        a = a.mean(1) + eye
        a = a / a.sum(-1, keepdim=True)
        R = a @ R
    m = R[:, 0, 1:]                      # jeton CLS vers les patchs
    side = int(m.shape[-1] ** 0.5)
    m = F.interpolate(m.reshape(-1, 1, side, side), size=x.shape[-2:], mode="bilinear")[:, 0]
    mn, mx = m.amin((1, 2), keepdim=True), m.amax((1, 2), keepdim=True)
    return logits.softmax(1), (m - mn) / (mx - mn + 1e-8)

def main(ckpt, n=8):
    tag = os.path.basename(ckpt).replace(".pt", "")
    ds = get_datasets()["test"]; labels = INFO["pathmnist"]["label"]
    idx = np.random.default_rng(1).choice(len(ds), n, replace=False)
    x = torch.stack([ds[int(i)][0] for i in idx]).to(dev)
    y = [int(ds[int(i)][1][0]) for i in idx]
    p, m = rollout(load(ckpt), x)
    fig, ax = plt.subplots(2, n, figsize=(2.2 * n, 5))
    for j in range(n):
        img = (x[j].cpu() * 0.5 + 0.5).permute(1, 2, 0).numpy()
        ax[0, j].imshow(img); ax[0, j].axis("off")
        ax[0, j].set_title(f"vrai: {labels[str(y[j])][:12]}", fontsize=7)
        ax[1, j].imshow(img); ax[1, j].imshow(m[j].cpu().numpy(), cmap="jet", alpha=0.5); ax[1, j].axis("off")
        ax[1, j].set_title(f"préd: {labels[str(int(p[j].argmax()))][:12]} {p[j].max():.2f}", fontsize=7)
    plt.tight_layout(); plt.savefig(f"results/attention_{tag}.png", dpi=150)

if __name__ == "__main__":
    main(sys.argv[1])