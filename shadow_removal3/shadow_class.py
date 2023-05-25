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
from models.class_module import Classifier



desc = "Pytorch implementation of Shadowremoval"
parser = argparse.ArgumentParser(description=desc)

parser.add_argument('--train_path', type=str,
                    default='/home/haoyu/Desktop/partical/ShadowNet_Data/train5/train', help='train_path')
parser.add_argument('--test_path', type=str,
                    default='/home/haoyu/Desktop/partical/ShadowNet_Data/test', help='test_path')
parser.add_argument('--result_dir', type=str,
                    default='/home/haoyu/Desktop/partical/shadow_removal3/output/ckpt')
parser.add_argument('--ckpt_dir', type=str,
                    default='/home/haoyu/Desktop/partical/shadow_removal3/output/ckpt')

parser.add_argument('--valid_ratio', type=float, default=0.2)

parser.add_argument('--max_epoch', type=int, default=200,
                    help='The number of training iterations')
parser.add_argument('--batch_size', type=int, default=32,
                    help='The size of batch size')


parser.add_argument('--lr', type=float, default=0.0001,
                    help='The learning rate')
parser.add_argument("--momentum", default=0.8, type=float,
                    metavar="F", help="Momentum.")
parser.add_argument("--weight_decay", default=2e-4,
                    type=float, metavar="F", help="Weight decay.")

parser.add_argument("--img_size", type=int, default=128,
                    help='The number of training iterations')


parser.add_argument('--seed', type=int, default=999)

args = parser.parse_args(args=[])

device = "cuda" if torch.cuda.is_available() else "cpu"
exp_time = datetime.today().strftime("%m-%d-%H")

output_folder = os.path.join(args.result_dir, exp_time)
os.makedirs(output_folder, exist_ok=True)

# ckpt_folder = os.path.join(args.ckpt_dir, exp_time)
# os.makedirs(ckpt_folder, exist_ok=True)

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

def train_valid_split(data_set, valid_ratio, seed):
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return train_set, valid_set

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
    def __img_preprocessing(self, img):
        image_array = np.array(img, dtype=np.float32)
        r, g, b = np.split(image_array, 3, axis=2)

        r_mean = np.mean(r)
        g_mean = np.mean(g)
        b_mean = np.mean(b)

        avg =  np.multiply(np.multiply(b_mean, g_mean), r_mean) ** (1.0/3)
        bCoef = avg/b_mean
        gCoef = avg/g_mean
        rCoef = avg/r_mean

        b = np.clip(b * bCoef, 0, 255)
        g = np.clip(g * gCoef, 0, 255)
        r= np.clip(r * rCoef, 0, 255)

        r_mean = np.mean(r)
        g_mean = np.mean(g)
        b_mean = np.mean(b)
        
        r_mean_new = np.mean(r[r > r_mean])
        g_mean_new = np.mean(g[g > g_mean])
        b_mean_new = np.mean(b[b > b_mean])
        new_coeff = 255 / max(r_mean_new, g_mean_new, b_mean_new)

        r = np.squeeze(r, axis = 2)
        g = np.squeeze(g, axis = 2)
        b = np.squeeze(b, axis = 2)

        image_array[:, :, 0] = r * new_coeff
        image_array[:, :, 1] = g * new_coeff
        image_array[:, :, 2] = b * new_coeff

        image_array = np.clip(image_array, 0, 255)

        img = Image.fromarray(np.uint8(image_array))
        return img
        
    def __len__(self):
        return len(self.folder_path)
    
    def __getitem__(self, index):
        fname = self.folder_path[index]
        im = Image.open(fname)
        im = self.__img_preprocessing(im)
        
        im = self.tfm(im)
        
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1
            
        return im, np.float32(label)
    


print("------------ Dataset --------------")
dataset = ShadowDataset(args.train_path)
train_dataset, valid_dataset = train_valid_split(dataset, args.valid_ratio, args.seed)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)

print(f"train_size : {len(train_dataset)}")
print(f"train_size : {len(valid_dataset)}")
print("-----------------------------------")
model = Classifier(img_size=args.img_size).to(device)

criterion = nn.BCELoss(reduction='mean') 
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) 
# optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

# summary(model,(3, 64, 64))

best_acc = 0

txt_path = os.path.join(output_folder, "log.txt")

for epoch in range(args.max_epoch):
    model.train()
    
    train_loss = []
    train_accs = []
    
    if epoch == 110:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay) 
        
    
    for batch in tqdm(train_loader):
        imgs, labels = batch
        
        labels = labels.to(device)
        
        logits = model(imgs.to(device))
        # print("logits:", logits.shape)
        # print("labels:", labels.shape)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        
        optimizer.step()
        
        acc = ((logits > 0.5) == labels).float().mean()
        
        train_loss.append(loss.item())
        train_accs.append(acc)
    
    train_loss = sum(train_loss)/len(train_loss)
    train_accs = sum(train_accs)/len(train_accs)
    
    print(f"[ Train | {epoch + 1:03d}/{args.max_epoch:03d} ] loss = {train_loss:.5f}, acc = {train_accs:.5f}")
    
    model.eval()
    
    valid_loss = []
    valid_accs = []
    
    for batch in tqdm(valid_loader):
        imgs, labels = batch
        labels = labels.to(device)
        
        with torch.no_grad():
            logits = model(imgs.to(device))
            
        loss = criterion(logits, labels)
        acc = ((logits > 0.5) == labels).float().mean()
        
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        
    valid_loss = sum(valid_loss)/len(valid_loss)
    valid_accs = sum(valid_accs)/len(valid_accs)
    
    print(f"[ Valid | {epoch + 1:03d}/{args.max_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_accs:.5f}")
    
    # scheduler.step(valid_accs)
    if(valid_accs > best_acc):
        with open(txt_path, "a") as f:
            print(f"[ Valid | {epoch + 1:03d}/{args.max_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_accs:.5f} -> new best")
            f.write(f"[ Valid | {epoch + 1:03d}/{args.max_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_accs:.5f} -> new best\n")
    else:
        with open(txt_path, "a") as f:
            print(f"[ Valid | {epoch + 1:03d}/{args.max_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_accs:.5f}")
            f.write(f"[ Valid | {epoch + 1:03d}/{args.max_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_accs:.5f}\n")
            
    if(valid_accs > best_acc):
        print(f"Best model found at epoch {epoch}, saving model")
        path = os.path.join(output_folder, "best.ckpt")
        torch.save(model.state_dict(), path) # only save best to prevent output memory exceed error
        best_acc = valid_accs
        