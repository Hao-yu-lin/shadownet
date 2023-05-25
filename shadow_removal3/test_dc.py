import random
import numpy as np
import torch
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from models.network import Discriminator, Generator, RhoClipper
from datetime import datetime
from tqdm.auto import tqdm
import cv2
import math

@staticmethod
def denorm(x):
    # mean=[0.485,0.456,0.406]
    # std=[0.229,0.224,0.225]
    # for i in range(3):
    #     x[i] = x[i] * std[i] + mean[i]
    return x * 0.5 + 0.5
    # return x

@staticmethod
def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

@staticmethod
def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

@staticmethod
def img_preprocess(img):
    image_array = np.array(img, dtype=np.float32)
    r, g, b = np.split(image_array, 3, axis=2)

    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)
    
    r_mean = np.mean(r[r > r_mean])
    g_mean = np.mean(g[g > g_mean])
    b_mean = np.mean(b[b > b_mean])

    avg =  np.multiply(np.multiply(b_mean, g_mean), r_mean) ** (1.0/3)
    bCoef = avg/b_mean
    gCoef = avg/g_mean
    rCoef = avg/r_mean

    b = np.clip(b * bCoef, 0, 255)
    g = np.clip(g * gCoef, 0, 255)
    r = np.clip(r * rCoef, 0, 255)

    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)
    
    r_mean_new = np.mean(r[r > r_mean])
    g_mean_new = np.mean(g[g > g_mean])
    b_mean_new = np.mean(b[b > b_mean])
    
    new_coeff1 = 255 / min(r_mean_new, g_mean_new, b_mean_new)
    new_coeff2 = 255 / min(r_mean, g_mean, b_mean)

    new_coeff = math.sqrt(new_coeff1 * new_coeff2)

    r = np.squeeze(r, axis = 2)
    g = np.squeeze(g, axis = 2)
    b = np.squeeze(b, axis = 2)

    image_array[:, :, 0] = r * new_coeff
    image_array[:, :, 1] = g * new_coeff
    image_array[:, :, 2] = b * new_coeff

    img = np.clip(image_array, 0, 255)

    return img


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
# output_dir1 = "/home/haoyu/Desktop/partical/ShadowNet_Data/train4/concat/" + exp_time
# os.makedirs(output_dir1, exist_ok=True)
output_dir2 = "/home/haoyu/Desktop/partical/ShadowNet_Data/0509_3"
os.makedirs(output_dir2, exist_ok=True)
test_dir = "/home/haoyu/Desktop/partical/ShadowNet_Data/test/test_shadow"
# test_dir = "/home/haoyu/Desktop/partical/ShadowNet_Data/test/test_shadow"


class CreatDataset(Dataset):
    def __init__(self, path):
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if (x.endswith(".png") or x.endswith(".jpg"))])
        self.tfm = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5),
        ])
        
        print(len(self.files))
        
    def __img_preprocessing(self, img):
        image_array = np.array(img, dtype=np.float32)
        r, g, b = np.split(image_array, 3, axis=2)

        r_mean = np.mean(r)
        g_mean = np.mean(g)
        b_mean = np.mean(b)
        
        r_mean = np.mean(r[r > r_mean])
        g_mean = np.mean(g[g > g_mean])
        b_mean = np.mean(b[b > b_mean])

        avg =  np.multiply(np.multiply(b_mean, g_mean), r_mean) ** (1.0/3)
        bCoef = avg/b_mean
        gCoef = avg/g_mean
        rCoef = avg/r_mean

        b = np.clip(b * bCoef, 0, 255)
        g = np.clip(g * gCoef, 0, 255)
        r = np.clip(r * rCoef, 0, 255)

        r_mean = np.mean(r)
        g_mean = np.mean(g)
        b_mean = np.mean(b)
        
        r_mean_new = np.mean(r[r > r_mean])
        g_mean_new = np.mean(g[g > g_mean])
        b_mean_new = np.mean(b[b > b_mean])
        
        new_coeff1 = 255 / min(r_mean_new, g_mean_new, b_mean_new)
        new_coeff2 = 255 / min(r_mean, g_mean, b_mean)

        new_coeff = math.sqrt(new_coeff1 * new_coeff2)

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
        return len(self.files)
    
    def __getitem__(self, index):
        fname = self.files[index]
        # name = fname.split("/")[-1].split(".")[0]
        
        img = Image.open(fname)
        img = img.convert('RGB')
        img = self.__img_preprocessing(img)
        img = self.tfm(img)
        
        return img, fname

test_set = CreatDataset(path = test_dir)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

param = torch.load("/home/haoyu/Desktop/GUI/parial-GUI/src/py_file/module_ckpt/05-08-16/train.pt")
model = Generator().to(device)
model.load_state_dict(param['genA2B'])
model.eval()



for idx, batch in enumerate(tqdm(test_loader)):
    imgs, fname = batch
    fname = fname[0].split("/")[-1].split(".")[0]
    with torch.no_grad():
        outputs, _, _ = model(imgs.to(device))
    
    
    # concat_img = np.concatenate((RGB2BGR(tensor2numpy(denorm(imgs[0]))), RGB2BGR(tensor2numpy(denorm(outputs[0])))), axis = 1)
    img = RGB2BGR(tensor2numpy(denorm(outputs[0])))
    img = img * 255.0
    # r, g, b = np.split(img, 3, axis=2)

    # r_mean = np.mean(r)
    # g_mean = np.mean(g)
    # b_mean = np.mean(b)
    
    # r_mean = np.mean(r[r > r_mean])
    # g_mean = np.mean(g[g > g_mean])
    # b_mean = np.mean(b[b > b_mean])

    # avg =  np.multiply(np.multiply(b_mean, g_mean), r_mean) ** (1.0/3)
    # bCoef = avg/b_mean
    # gCoef = avg/g_mean
    # rCoef = avg/r_mean

    # b = np.clip(b * bCoef, 0, 255)
    # g = np.clip(g * gCoef, 0, 255)
    # r = np.clip(r * rCoef, 0, 255)

    # r_mean = np.mean(r)
    # g_mean = np.mean(g)
    # b_mean = np.mean(b)
    
    # r_mean_new = np.mean(r[r > r_mean])
    # g_mean_new = np.mean(g[g > g_mean])
    # b_mean_new = np.mean(b[b > b_mean])
    
    # new_coeff1 = 255 / min(r_mean_new, g_mean_new, b_mean_new)
    # new_coeff2 = 255 / min(r_mean, g_mean, b_mean)

    # new_coeff = math.sqrt(new_coeff1 * new_coeff2)

    # r = np.squeeze(r, axis = 2)
    # g = np.squeeze(g, axis = 2)
    # b = np.squeeze(b, axis = 2)

    # img[:, :, 0] = r * new_coeff
    # img[:, :, 1] = g * new_coeff
    # img[:, :, 2] = b * new_coeff

    # img = np.clip(img, 0, 255)

    # img = np.uint8(img)
    # count = 10000 + idx
    # cv2.imwrite(os.path.join(output_dir1, f"{fname}_concate.png"), concat_img * 255.0)
    cv2.imwrite(os.path.join(output_dir2, f"{fname}.png"), img)
    
    
    