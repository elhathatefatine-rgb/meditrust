import time, copy, numpy as np, torch, timm, gradio as gr
import torchvision.transforms as T
import matplotlib.cm as cm
from medmnist import INFO
from src.train import MODELS
from src.explain import rollout

torch.set_grad_enabled(False)
LABELS = [INFO["pathmnist"]["label"][str(i)] for i in range(9)]
CKPTS = {"Standard": "checkpoints/fed_k5_a0.5.pt",
         "Renforcé (adversarial)": "checkpoints/fed_k5_a0.5_adv2.pt"}
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

def load(path):
    m = timm.create_model(MODELS["vit"], pretrained=False, num_classes=9)
    m.load_state_dict(torch.load(path, map_location="cpu")); m.eval()
    int8 = torch.ao.quantization.quantize_dynamic(copy.deepcopy(m), {torch.nn.Linear}, dtype=torch.qint8)
    for b in m.blocks: b.attn.fused_attn = False
    return m, int8

NETS = {k: load(p) for k, p in CKPTS.items()}

def timed(model, x, runs=5):
    model(x); t = time.perf_counter()
    for _ in range(runs): out = model(x)
    return out.softmax(1)[0], (time.perf_counter() - t) / runs * 1000

def predict(img, which, tau):
    if img is None: return None, None, "", ""
    img = img.convert("RGB"); x = tf(img).unsqueeze(0)
    fp32, int8 = NETS[which]
    p, t32 = timed(fp32, x); q, t8 = timed(int8, x)
    _, m = rollout(fp32, x)
    base = np.asarray(img.resize((224, 224))) / 255.0
    overlay = (0.5 * base + 0.5 * cm.jet(m[0].numpy())[..., :3]).clip(0, 1)
    conf = p.max().item()
    status = (f"⚠️ À vérifier par un médecin (confiance {conf:.2f} < seuil {tau:.2f})" if conf < tau
              else f"✅ Confiance suffisante ({conf:.2f} ≥ seuil {tau:.2f})")
    timing = f"FP32 : {t32:.0f} ms ({LABELS[p.argmax()]})  |  INT8 : {t8:.0f} ms ({LABELS[q.argmax()]})"
    return {LABELS[i]: float(p[i]) for i in range(9)}, (overlay * 255).astype("uint8"), status, timing

demo = gr.Interface(
    fn=predict,
    inputs=[gr.Image(type="pil", label="Image histologique"),
            gr.Radio(list(CKPTS), value="Standard", label="Modèle"),
            gr.Slider(0.5, 1.0, value=0.9, step=0.01, label="Seuil de confiance")],
    outputs=[gr.Label(num_top_classes=3, label="Diagnostic"),
             gr.Image(label="Carte d'attention (rollout)"),
             gr.Textbox(label="Statut"), gr.Textbox(label="Temps d'inférence")],
    title="MediTrust : prototype de recherche",
    description="Prototype sur PathMNIST (histologie colorectale). Ne constitue pas un outil de diagnostic médical.")

if __name__ == "__main__":
    demo.launch()