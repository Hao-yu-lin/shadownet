import random
import numpy as np
import torch
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from models.network import Discriminator, Generator, RhoClipper
from models.class_module import Classifier
from datetime import datetime
from tqdm.auto import tqdm
import cv2
import math

def denorm(x):
    return x * 0.5 + 0.5

def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

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
output_dir1 = "/home/haoyu/Desktop/partical/ShadowNet_Data/train4/concat/" + exp_time
os.makedirs(output_dir1, exist_ok=True)
output_dir2 = "/home/haoyu/Desktop/partical/ShadowNet_Data/train4/output/" + exp_time
os.makedirs(output_dir2, exist_ok=True)
test_dir = "/home/haoyu/Desktop/partical/ShadowNet_Data/test/600dpi"

# test_dir = "/home/haoyu/Desktop/partical/ShadowNet_Data/test/test_shadow"

def combine_patches(patches, w, h):
    num_cols = math.ceil(w/patches[0].size[0])
    num_rows = math.ceil(h/patches[0].size[1])
    
    new_im = Image.new('RGB', (num_cols * patches[0].size[0], num_rows * patches[0].size[1]))
    for i in range(num_rows):
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx < len(patches):
                x = j * patches[0].size[0]
                y = i * patches[0].size[1]
                new_im.paste(patches[idx], (x, y))
    return new_im[:w, :h]

class CreatDataset(Dataset):
    def __init__(self, path, image_size=128):
        self.image_path = path
        self.image_size = image_size
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if (x.endswith(".png") or x.endswith(".jpg"))])
        self.tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ])
        self.w = None
        self.h = None

        
        
    def __len__(self):
        return len(self.files)
    
    def origin_size(self):
        return self.w, self.h
    
    def __getitem__(self, index):
        fname = self.files[index]
        img = Image.open(fname).convert('RGB')
        
        self.w, self.h = img.size

        num_rows = self.h // self.image_size
        num_cols = self.w // self.image_size
       
        
        patches = []
        for i in range(num_rows):
            for j in range(num_cols):
                x = j * self.image_size
                y = i * self.image_size
                
                patch = img.crop((x, y, x + self.image_size, y + self.image_size))
                patch = self.tfm(patch)
                patches.append(patch)
                
        return patches

test_set = CreatDataset(path = test_dir)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

param = torch.load("/home/haoyu/Desktop/partical/shadow_removal2/finish_copy/ckpt/train.pt")
model = Generator().to(device)
model.load_state_dict(param['genA2B'])
model.eval()



# print(test_set.origin_size)
w, h = test_set.origin_size

patches = []
for batch in tqdm(test_loader):
    imgs = batch
    
    fname = fname[0].split("/")[-1].split(".")[0]
    with torch.no_grad():
        # logitces = class_model(imgs.to(device))
        # if(logitces > 0.5):
        #     outputs, _, _ = model(imgs.to(device))
        #     outputs = outputs[0]
        # else:
        #     outputs = imgs.to(device)
        outputs, _, _ = model(imgs.to(device))
    
    # concat_img = np.concatenate((RGB2BGR(tensor2numpy(denorm(imgs[0]))), RGB2BGR(tensor2numpy(denorm(outputs[0])))), axis = 1)
    img = RGB2BGR(tensor2numpy(denorm(outputs[0])))
    patches.append(img * 255.0)
    # cv2.imwrite(os.path.join(output_dir1, f"{fname}_concate.png"), concat_img * 255.0)
    # cv2.imwrite(os.path.join(output_dir2, f"{fname}.png"), img * 255.0)
    new_img = combine_patches(patches, w, h)
    cv2.imwrite(os.path.join(output_dir2, f"test.png"), new_img)




# # class_model = Classifier(img_size=128).to(device)
# # class_model.load_state_dict(torch.load("/home/haoyu/Desktop/partical/shadow_removal3/output/ckpt/03-12-12_128/best.ckpt"))
# # class_model.eval()

# with torch.no_grad():
#         # logitces = class_model(imgs.to(device))
#         # if(logitces > 0.5):
#         #     outputs, _, _ = model(imgs.to(device))
#         #     outputs = outputs[0]
#         # else:
#         #     outputs = imgs.to(device)
#         outputs, _, _ = model(imgs.to(device