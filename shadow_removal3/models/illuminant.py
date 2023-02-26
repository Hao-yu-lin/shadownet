import cv2
import numpy as np
import math
import torch
import time
from multiprocessing import Pool
from torchvision import transforms

def denorm(img):
    img = img * 0.5 + 0.5
    return img


def ill_calcuate(img):
    # img = torch.squeeze(img, dim = 0)
    
    img = img.permute(1, 2, 0)
    img = denorm(img)
    img = img * 255
    
    nel = img.shape[0] * img.shape[1]
    
    blue = img[:, :, 0].detach().cpu().numpy().astype(np.uint8)
    blue = blue.astype(np.float64)
    # blue = img[:, :, 0].detach().cpu().numpy()
    
    green = img[:, :, 1].detach().cpu().numpy().astype(np.uint8)
    green = green.astype(np.float64)
    
    # green = img[:, :, 1].detach().cpu().numpy()
    
    red = img[:, :, 2].detach().cpu().numpy().astype(np.uint8)
    red = red.astype(np.float64)
    
    # red = img[:, :, 2].detach().cpu().numpy()

    
    blue[blue == 0] = 1
    green[green == 0] = 1
    red[red == 0] = 1
    
    # print(blue)
    # stage_time1 = time.time()
    
    div = np.multiply(np.multiply(blue, green), red) ** (1.0/3)
    
    cb = (blue/div)
    cg = (green/div)
    cr = (red/div)
    

    log_b = np.atleast_3d(np.log(cb))
    log_g = np.atleast_3d(np.log(cg))
    log_r = np.atleast_3d(np.log(cr))
    
    
    # log space
    rho = np.concatenate((log_r, log_g, log_b), axis=2)

    # U = [v1, v2] v1 = [1/sqrt(2), -1/sqrt(2), 0], v2 = [1/sqrt(6), 1/sqrt(6), -2/sqrt(6)]
    U = [[1/math.sqrt(2), -1/math.sqrt(2), 0],
        [1/math.sqrt(6), 1/math.sqrt(6), -2/math.sqrt(6)]]
    U = np.array(U)  # eigens

    # X = [X1, X2]
    X = np.dot(rho, U.T)
    
    rho_ti = np.dot(X, U)
    c_ti = np.exp(rho_ti)
    sum_ti = np.sum(c_ti, axis=2)
    sum_ti = sum_ti.reshape(c_ti.shape[0], c_ti.shape[1], 1)
    r_ti = c_ti/sum_ti
    
    r_ti2 = 1-r_ti[:, :, 0]
    # r_ti2 = 255.0 * r_ti2
    # r_ti2[np.isnan(r_ti2)] = 0
    
    r_ti2 = transforms.ToTensor()(r_ti2)
    # r_ti2 = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(r_ti2)
    # r_ti2 = norm(r_ti2)
    
    # r_ti2 = r_ti2.permute(2, 0, 1)
    r_ti2 = torch.unsqueeze(r_ti2, dim=0)
    # r_ti2 = intr_tfm(r_ti2)
    # stage_time5 = time.time()
    
    return r_ti2
def illuminant(img):
    new_ill = None
    
    for batch in img:
        if new_ill is None:
            new_ill = ill_calcuate(batch)
        else:
            temp_ill = ill_calcuate(batch)
            new_ill = torch.cat([new_ill, temp_ill], dim=0)
    
    return new_ill


