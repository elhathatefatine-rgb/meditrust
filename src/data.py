import os
import medmnist
from medmnist import INFO
from torch.utils.data import DataLoader
import torchvision.transforms as T

def get_datasets(flag="pathmnist", size=64, img=224, root="data"):
    os.makedirs(root, exist_ok=True)
    DS = getattr(medmnist, INFO[flag]["python_class"])
    tf = T.Compose([T.Resize(img), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    return {s: DS(split=s, transform=tf, download=True, size=size, root=root)
            for s in ["train", "val", "test"]}
    
def get_loaders(bs=64, **kw):
    ds = get_datasets(**kw)
    return {s: DataLoader(d, batch_size=bs, shuffle=(s == "train"), num_workers=0)
            for s, d in ds.items()}