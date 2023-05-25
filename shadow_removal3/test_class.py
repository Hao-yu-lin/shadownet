import os
import argparse
import random
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from tqdm.auto import tqdm
from datetime import datetime
from PIL import Image
from torchsummary import summary
from timm.models.layers import to_2tuple
from models.class_module import Classifier



desc = "Pytorch implementation of Shadowremoval"
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('--test_path', type=str,
                    default='/home/haoyu/Desktop/partical/ShadowNet_Data/train5/train', help='test_path')
parser.add_argument('--result_dir', type=str,
                    default='/home/haoyu/Desktop/partical/shadow_removal3/output/ckpt')
parser.add_argument('--ckpt_dir', type=str,
                    default='/home/haoyu/Desktop/partical/shadow_removal3/output/ckpt')

parser.add_argument("--img_size", type=int, default=128,
                    help='The number of training iterations')


parser.add_argument('--max_epoch', type=int, default=200,
                    help='The number of training iterations')
parser.add_argument('--batch_size', type=int, default=32,
                    help='The size of batch size')


parser.add_argument('--seed', type=int, default=999)

args = parser.parse_args(args=[])

device = "cuda" if torch.cuda.is_available() else "cpu"
exp_time = datetime.today().strftime("%m-%d-%H")

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(args.seed)

class ShadowDataset(Dataset):
    def __init__(self, path, tfm = None):
        super(ShadowDataset).__init__()
        self.folder_path = sorted([
            os.path.join(path, x) for x in os.listdir(path) if x.endswith(".jpg") or x.endswith(".png")
            ])
        
        print(f"Total img data {len(self.folder_path)}")
        
        if tfm is None:
            self.tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ])
        else:
            self.tfm = tfm
        
    def __len__(self):
        return len(self.folder_path)
    
    def __getitem__(self, index):
        fname = self.folder_path[index]
        im = Image.open(fname)
        im = self.tfm(im)
        
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1
            
        return im, np.float32(label)
    
dataset = ShadowDataset(args.test_path)
test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
print("------------ Dataset --------------")
print(f"train_size : {len(dataset)}")
print("-----------------------------------")

model = Classifier(img_size=args.img_size).to(device)
model.load_state_dict(torch.load("/home/haoyu/Desktop/partical/shadow_removal3/output/ckpt/03-12-11/best.ckpt"))
model.eval()

test_accs = []

for data in tqdm(test_loader):
    imgs, labels = data
    labels = labels.to(device)
    
    with torch.no_grad():
        logits = model(imgs.to(device))
        
    acc = ((logits > 0.5) == labels).float().mean()
    
    test_accs.append(acc)
    
test_accs = sum(test_accs)/len(test_accs)
print(f"acc = {test_accs:.5f}")