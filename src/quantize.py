import sys, os, io, copy, json, time, numpy as np, torch, timm
from torch.utils.data import DataLoader, Subset
from src.data import get_datasets
from src.train import MODELS

torch.set_grad_enabled(False)

def size_mb(model):
    buf = io.BytesIO(); torch.save(model.state_dict(), buf); return buf.tell() / 1e6

def accuracy(model, loader):
    model.eval(); ok = n = 0
    for x, y in loader:
        ok += (model(x).argmax(1) == y.squeeze(1).long()).sum().item(); n += len(y)
    return ok / n

def latency_ms(model, runs=30, reps=7, warmup=10):
    model.eval(); x = torch.randn(1, 3, 224, 224)
    for _ in range(warmup): model(x)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        for _ in range(runs): model(x)
        ts.append((time.perf_counter() - t) / runs * 1000)
    return float(np.median(ts))

def main(ckpt, name="vit", n=2000):
    tag = os.path.basename(ckpt).replace(".pt", "")
    ds = get_datasets()["test"]
    idx = np.random.default_rng(0).choice(len(ds), n, replace=False).tolist()
    loader = DataLoader(Subset(ds, idx), batch_size=32)
    fp32 = timm.create_model(MODELS[name], pretrained=False, num_classes=9)
    fp32.load_state_dict(torch.load(ckpt, map_location="cpu"))
    int8 = torch.ao.quantization.quantize_dynamic(copy.deepcopy(fp32), {torch.nn.Linear}, dtype=torch.qint8)
    res = {}
    for k, m in [("fp32", fp32), ("int8", int8)]:
        res[k] = {"size_mb": size_mb(m), "latency_ms": latency_ms(m), "acc": accuracy(m, loader)}
        print(k, res[k])
    json.dump(res, open(f"results/quant_{tag}.json", "w"), indent=2)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "vit")