from PIL import Image
import medmnist
from app import predict, LABELS

d = medmnist.PathMNIST(split="test", size=64, root="data")
for i in range(0, 80, 10):
    probs, _, _, _ = predict(Image.open(f"examples/img{i}.png"), "Standard", 0.9)
    top = max(probs, key=probs.get)
    flag = "A VERIFIER" if probs[top] < 0.9 else "ok"
    print(f"img{i}: vrai={LABELS[int(d[i][1][0])]} | pred={top} ({probs[top]:.2f}) | {flag}")
    