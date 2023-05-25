import random
import numpy as np
import torch
import os
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from models.network import Discriminator, Generator, RhoClipper
from datetime import datetime
from tqdm.auto import tqdm
import cv2
import math
from models.class_module import Classifier

def denorm(x):
    return x * 0.5 + 0.5

def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0) # H X W X C

def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


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
    
same_seeds(999)
device = "cuda" if torch.cuda.is_available() else "cpu"
exp_time = datetime.today().strftime("%m-%d-%H")
output_dir1 = "/home/haoyu/Desktop/partical/ShadowNet_Data/test/600dpi/concat/" + exp_time
os.makedirs(output_dir1, exist_ok=True)
output_dir2 = "/home/haoyu/Desktop/partical/ShadowNet_Data/test/600dpi/output/" + exp_time
os.makedirs(output_dir2, exist_ok=True)
test_dir = "/home/haoyu/Desktop/partical/ShadowNet_Data/test/600dpi"

# test_dir = "/home/haoyu/Desktop/partical/ShadowNet_Data/test/test_shadow"

def combine_patches(patches, origin_size, img_size=128):
    # print(len(patches))
    # print(len(patches[0]))
    patches = np.array(patches)
    # print(patches.shape)
    origin_h, origin_w = origin_size
    n_h = (int(origin_h/img_size) if origin_h % img_size == 0 else int(origin_h/img_size) + 1)
    n_w = (int(origin_w/img_size) if origin_w % img_size == 0 else int(origin_w/img_size) + 1)
    new_concat = []
    for i in range(n_w):
        new = np.concatenate(patches[i*n_h:i*n_h+n_h], axis=1)
        new_concat.append(new)
    new_concat = np.array(new_concat)
    new_concat = np.concatenate(new_concat, axis=0)
    new_img = new_concat[:origin_w, :origin_h]
    # print(new_img.shape)
    return new_img


    

class CreatDataset(Dataset):
    def __init__(self, path, img_size=128):
        self.image_path = path
        self.img_size = img_size
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if (x.endswith(".png") or x.endswith(".jpg"))])
        self.img_files = self.__split__()
        
        self.tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ])
        
        
    def __split__(self):
        img = Image.open(self.files[0]).convert('RGB')
        origin_h, origin_w = img.size
        self.origin_size = (origin_h, origin_w)
    
        n_h = (int(origin_h/self.img_size) if origin_h % self.img_size == 0 else int(origin_h/self.img_size) + 1)
        n_w = (int(origin_w/self.img_size) if origin_w % self.img_size == 0 else int(origin_w/self.img_size) + 1)


        h = n_h * self.img_size - origin_h
        w = n_w * self.img_size - origin_w

        new_img = ImageOps.expand(img, (0, 0, w, h), fill="white")

        patches = []

        for i in range(n_w):
            for j in range(n_h):
                x = j * self.img_size
                y = i * self.img_size
                
                patch = new_img.crop((x, y, x + self.img_size, y + self.img_size))
                patches.append(patch)
        return patches
    
        
    def __len__(self):
        return len(self.img_files)
    
    def __imgsize__(self):
        return self.origin_size
    
    def __getitem__(self, index):
        img = self.img_files[index]
        img = self.tfm(img)
        
                
        return img

test_set = CreatDataset(path = test_dir, img_size=128)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

param = torch.load("/home/haoyu/Desktop/partical/shadow_removal3/output/ckpt/03-14-04/train.pt")
model = Generator().to(device)
model.load_state_dict(param['genA2B'])
model.eval()

origin_size = test_set.__imgsize__()

class_model = Classifier(img_size=128).to(device)
class_model.load_state_dict(torch.load("/home/haoyu/Desktop/partical/shadow_removal3/output/ckpt/03-12-12_128/best.ckpt"))
class_model.eval()


patches = []
for batch in tqdm(test_loader):
    imgs = batch
    b, _, _, _ = imgs.shape
   
    
    # fname = fname[0].split("/")[-1].split(".")[0]
    with torch.no_grad():
        # outputs, _, _ = model(imgs.to(device))
        logitces = class_model(imgs.to(device))
        if(logitces > 0.5):
            outputs, _, _ = model(imgs.to(device))
            # outputs = outputs[0]
        else:
            outputs = imgs.to(device)
    
    # print(outputs.shape)
    for i in range(b):
        tmp_img = RGB2BGR(tensor2numpy(denorm(outputs[i])))
        patches.append(tmp_img * 255.0)


concat_img = combine_patches(patches, origin_size)
cv2.imwrite(f"/home/haoyu/Desktop/partical/tmp_img/A1-390_600dpi-co_class.png", concat_img)
    



