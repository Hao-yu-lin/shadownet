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

def task(Y, bin_width):
    temp = []
    mean = np.mean(Y[:, :])
    std = np.std(Y[:, :])
    

    comp1 = mean-3*std
    comp2 = mean+3*std
    
    # CI 99.7%
    for j in range(Y.shape[0]):
        for k in range(Y.shape[1]):
            if Y[j][k] > comp1 and Y[j][k] < comp2:
                temp.append(Y[j][k])
    
    
    nbins = round((max(temp)-min(temp))/bin_width)+1

    hist, bins = np.histogram(temp, bins=nbins)
    hist = filter(lambda var1: var1 != 0, hist)
    hist1 = np.array([float(var) for var in hist])
    hist1 = hist1/sum(hist1)
    


    return -1*sum(np.multiply(hist1, np.log2(hist1)))

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
    
    # cb[np.isnan(cb)] = 0
    # cg[np.isnan(cg)] = 0
    # cr[np.isnan(cr)] = 0
    
    # print(np.all(np.isnan(cb) == False))
    # print(np.all(np.isnan(cb) == False))
    # print(np.all(np.isnan(cb) == False))
    
    

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
    # stage_time2 = time.time()
    # e_t = np.zeros((2, 181))
    # for theta in range(181):
    #     # rad = pi * deg / 180.0
    #     e_t[0][theta] = math.cos(theta * math.pi/180.0)
    #     e_t[1][theta] = math.sin(theta * math.pi/180.0)

    # # x_1 cos + x_2 sin
    # Y = np.dot(X, e_t)

    
    # bin_width = np.zeros(181)
    # for theta in range(181):
    #     bin_width[theta] = (3.5 * np.std(Y[:, :, theta])) * (nel ** (-1.0/3))
        
    # stage_time3 = time.time()

    # entropy = []
    # with Pool(14) as mp_pool:
    #     items = [(Y[:,:,theta], bin_width[theta]) for theta in range(181)]
    #     for result in mp_pool.starmap(task, items):
    #         entropy.append(result)
    # # for theta in range(181):
    #     stage_time1 = time.time()
    #     temp = []
    #     mean = np.mean(Y[:, :, theta])
    #     std = np.std(Y[:, :, theta])
        

    #     comp1 = mean-3*std
    #     comp2 = mean+3*std
    #     stage_time2 = time.time()
        
    #     # CI 99.7%
    #     for j in range(Y.shape[0]):
    #         for k in range(Y.shape[1]):
    #             if Y[j][k][theta] > comp1 and Y[j][k][theta] < comp2:
    #                 temp.append(Y[j][k][theta])
        
    #     # print(temp.shape)
    #     stage_time3 = time.time()
        
    #     nbins = round((max(temp)-min(temp))/bin_width[theta])
    #     # nbins = 255
    #     #  cal prob
    #     print(nbins)
    #     hist, bins = np.histogram(temp, bins=nbins)
    #     hist = filter(lambda var1: var1 != 0, hist)
    #     hist1 = np.array([float(var) for var in hist])
    #     hist1 = hist1/sum(hist1)
        
    #     stage_time4 = time.time()

    #     entropy.append(-1*sum(np.multiply(hist1, np.log2(hist1))))
        
        
    #     print("stage2:", stage_time2 - stage_time1)
    #     print("stage3:", stage_time3 - stage_time2)
    #     print("stage4:", stage_time4 - stage_time3)
    #     # print("stage5:", stage_time5 - stage_time4)
    
    #     # print(" ok ")
    
    # min_angle = entropy.index(min(entropy))
    # stage_time4 = time.time()

    # e_t = np.array([math.cos(min_angle * math.pi/180.0),
    #             math.sin(min_angle * math.pi/180.0)])
    

    # cos^2 + sin^2
    # p_theta = np.dot(e_t.T, e_t)
    # X_theta = X * p_theta

    rho_ti = np.dot(X, U)
    c_ti = np.exp(rho_ti)
    sum_ti = np.sum(c_ti, axis=2)
    sum_ti = sum_ti.reshape(c_ti.shape[0], c_ti.shape[1], 1)
    r_ti = c_ti/sum_ti
    r_ti2 = r_ti
    
    # r_ti2[np.isnan(r_ti2)] = 0
    
    r_ti2 = transforms.ToTensor()(r_ti2)
    r_ti2 = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(r_ti2)
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


# def illuminant(img):
#     img = torch.squeeze(img, dim = 0)
    
#     img = img.permute(1, 2, 0)
#     img = img * 255
    
#     nel = img.shape[0] * img.shape[1]
    
#     blue = img[:, :, 0].to(dtype=torch.uint8)
#     blue = blue.to(dtype=torch.float64)
       
#     green = img[:, :, 1].to(dtype=torch.uint8)
#     green = green.to(dtype=torch.float64)
    
#     red = img[:, :, 2].to(dtype=torch.uint8)
#     red = red.to(dtype=torch.float64)
       
#     blue[blue == 0] = 1
#     green[green == 0] = 1
#     red[red == 0] = 1
    
#     # print(blue)
    
#     div = torch.multiply(torch.multiply(blue, green), red) ** (1.0/3)
    
#     cb = (blue/div)
#     cg = (green/div)
#     cr = (red/div)
    
#     # cb[np.isnan(cb)] = 0
#     # cg[np.isnan(cg)] = 0
#     # cr[np.isnan(cr)] = 0
    
#     # print(torch.all(torch.isnan(cb) == False))
#     # print(torch.all(torch.isnan(cb) == False))
#     # print(torch.all(torch.isnan(cb) == False))
    
    

#     log_b = torch.atleast_3d(torch.log(cb))
#     log_g = torch.atleast_3d(torch.log(cg))
#     log_r = torch.atleast_3d(torch.log(cr))
    
    
#     # log space
#     rho = torch.cat((log_r, log_g, log_b), axis=2)

#     # U = [v1, v2] v1 = [1/sqrt(2), -1/sqrt(2), 0], v2 = [1/sqrt(6), 1/sqrt(6), -2/sqrt(6)]
#     U = [[1/math.sqrt(2), -1/math.sqrt(2), 0],
#         [1/math.sqrt(6), 1/math.sqrt(6), -2/math.sqrt(6)]]
    
#     U = torch.Tensor(U).cuda().to(dtype=torch.float64) # eigens

#     # X = [X1, X2]
#     X = rho @ U.T

#     e_t = torch.zeros((2, 181)).cuda()
#     for theta in range(181):
#         # rad = pi * deg / 180.0
#         e_t[0][theta] = math.cos(theta * math.pi/180.0)
#         e_t[1][theta] = math.sin(theta * math.pi/180.0)

#     # x_1 cos + x_2 sin

#     e_t = e_t.to(dtype=torch.float64)
#     Y = X @ e_t

#     bin_width = torch.zeros(181).cuda()
#     for theta in range(181):
#         bin_width[theta] = (3.5 * torch.std(Y[:, :, theta])) * (nel ** (-1.0/3))
        
#     bin_width = bin_width.to(dtype=torch.float64)
#     entropy = []
#     for theta in range(181):
#         temp = []
#         mean = torch.mean(Y[:, :, theta])
#         std = torch.std(Y[:, :, theta])

#         comp1 = mean-3*std
#         comp2 = mean+3*std
#         # CI 99.7%
#         for j in range(Y.shape[0]):
#             for k in range(Y.shape[1]):
#                 if Y[j][k][theta] > comp1 and Y[j][k][theta] < comp2:
#                     temp.append(Y[j][k][theta])
        
#         nbins = torch.round((max(temp)-min(temp))/bin_width[theta])
#         # nbins = 255
#         #  cal prob
#         temp = torch.Tensor(temp)
#         hist = torch.histc(temp, bins=nbins.to(torch.int64))
#         hist = filter(lambda var1: var1 != 0, hist)
#         hist1 = torch.Tensor([float(var) for var in hist])
#         hist1 = hist1/torch.sum(hist1)
        
#         entropy.append(-1*torch.sum(hist1 @ torch.log2(hist1)))

#     min_angle = entropy.index(min(entropy))

#     e_t = [math.cos(min_angle * math.pi/180.0),
#                 math.sin(min_angle * math.pi/180.0)]
    
#     e_t = torch.Tensor(e_t).cuda().to(dtype=torch.float64)
    

#     # cos^2 + sin^2
#     # p_theta = np.dot(e_t.T, e_t)
#     p_theta = e_t.T @ e_t
    
#     X_theta = X * p_theta

#     rho_ti = X_theta @ U
#     # rho_ti = np.dot(X_theta, U)
#     print(" ----- ok ------")
#     c_ti = np.exp(rho_ti)
#     sum_ti = np.sum(c_ti, axis=2)
#     sum_ti = sum_ti.reshape(c_ti.shape[0], c_ti.shape[1], 1)
#     r_ti = c_ti/sum_ti
#     r_ti2 = 255 * r_ti
    
#     # r_ti2[np.isnan(r_ti2)] = 0
    
#     r_ti2 = torch.from_numpy(r_ti2)
#     r_ti2 = r_ti2.permute(2, 0, 1)
#     r_ti2 = torch.unsqueeze(r_ti2, dim=0)
#     # print("--------")
    
#     # print(torch.all(torch.isnan(r_ti2) == False))
    
#     return r_ti2