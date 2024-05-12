import os, torch, json, shutil, numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torchvision import datasets, models
import torch.nn.functional as F
from glob import glob; from PIL import Image
from torch.utils.data import random_split, Dataset, DataLoader
from torchvision import transforms as T
import random
torch.manual_seed(42)

class CustomDataset(Dataset):
    
    def __init__(self, root, data, transformations = None):
        
        self.transformations = transformations
        self.im_paths = sorted(glob(f"{root}/{data}/*/*"))
        json_data = json.load(open(f"{root}/cat_to_name.json"))
        self.cls_names = {}
        
        for idx, (key, value) in enumerate(json_data.items()): self.cls_names[int(key) - 1] = value
        
        self.cls_counts, count = {}, 0
        for idx, im_path in enumerate(self.im_paths):
            class_name = self.cls_names[int(self.get_class(im_path)) - 1]
            if class_name not in self.cls_counts: self.cls_counts[class_name] = 1
            else: self.cls_counts[class_name] += 1
        
    def get_class(self, path): return os.path.basename(os.path.dirname(path))
    
    def __len__(self): return len(self.im_paths)

    def __getitem__(self, idx):
        
        im_path = self.im_paths[idx]
        im = Image.open(im_path).convert("RGB")
        gt = int(self.get_class(im_path)) - 1
        
        if self.transformations is not None: im = self.transformations(im)
        
        return im, gt
    
def get_dls(root, transformations, bs, ns = 4):
    
    tr_ds = CustomDataset(root = root, data = "train", transformations = transformations)
    vl_ds = CustomDataset(root = root, data = "valid", transformations = transformations)
    ts_ds = CustomDataset(root = root, data = "test",  transformations = transformations)
    
    tr_dl, val_dl, ts_dl = DataLoader(tr_ds, batch_size = bs, shuffle = True, num_workers = ns), DataLoader(vl_ds, batch_size = bs, shuffle = False, num_workers = ns), DataLoader(ts_ds, batch_size = 1, shuffle = False, num_workers = ns)
    
    return tr_dl, val_dl, ts_dl, tr_ds.cls_names

def tensor_2_im(t, t_type = "rgb"):
    
    gray_tfs = T.Compose([T.Normalize(mean = [ 0.], std = [1/0.5]), T.Normalize(mean = [-0.5], std = [1])])
    rgb_tfs = T.Compose([T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]), T.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ])])
    
    invTrans = gray_tfs if t_type == "gray" else rgb_tfs 
    
    return (invTrans(t) * 255).detach().squeeze().cpu().permute(1,2,0).numpy().astype(np.uint8) if t_type == "gray" else (invTrans(t) * 255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)

def visualize(data, n_ims, rows, cmap = None, cls_names = None):
    
    assert cmap in ["rgb", "gray"], "Specify whether the picture is black and white or color!"
    if cmap == "rgb": cmap = "viridis"
    
    plt.figure(figsize = (20, 10))
    indekslar = [random.randint(0, len(data) - 1) for _ in range(n_ims)]
    for idx, indeks in enumerate(indekslar):
        
        im, gt = data[indeks]
        # Start plot
        plt.subplot(rows, n_ims // rows, idx + 1)
        if cmap: 
            plt.imshow(tensor_2_im(im, cmap), cmap=cmap)
        else: 
            plt.imshow(tensor_2_im(im))
        plt.axis('off')
        if cls_names is not None: 
            plt.title(f"GT -> {cls_names[gt]}")
        else: 
            plt.title(f"GT -> {gt}")
            





if __name__ == '__main__':
    root = "data/oxford-102-flower-pytorch/flower_data"
    mean, std, im_size = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 224
    tfs = T.Compose([T.Resize((im_size, im_size)), T.ToTensor(), T.Normalize(mean = mean, std = std)])
    tr_dl, val_dl, ts_dl, classes = get_dls(root = root, transformations = tfs, bs = 32)

    print(len(tr_dl)); print(len(val_dl)); print(len(ts_dl)); print(classes)

    visualize(tr_dl.dataset, 20, 4, "rgb", list(classes.values()))