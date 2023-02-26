# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# import os
# import math
# import copy
# from multiprocessing import Pool


# def illuminant(img_path):
#     img_name = img_path.split("/")[-1].split(".")[0]

#     try:
#         img = cv2.imread(img_path)
#         img = np.float64(img)
#         orig_img = copy.deepcopy(img)
#         blue, green, red = cv2.split(img)

#         # fix blue, green, red == 0
#         blue[blue == 0] = 1
#         green[green == 0] = 1
#         red[red == 0] = 1


#         # cal geometric mean of b、g、r => ck
#         div = np.multiply(np.multiply(blue, green), red) ** (1.0/3)

#         cb = (blue/div)
#         cg = (green/div)
#         cr = (red/div)

#         log_b = np.atleast_3d(np.log(cb))
#         log_g = np.atleast_3d(np.log(cg))
#         log_r = np.atleast_3d(np.log(cr))

#         # log space
#         rho = np.concatenate((log_r, log_g, log_b), axis=2)

#         # U = [v1, v2] v1 = [1/sqrt(2), -1/sqrt(2), 0], v2 = [1/sqrt(6), 1/sqrt(6), -2/sqrt(6)]
#         U = [[1/math.sqrt(2), -1/math.sqrt(2), 0],
#             [1/math.sqrt(6), 1/math.sqrt(6), -2/math.sqrt(6)]]
#         U = np.array(U)  # eigens

#         # X = [X1, X2]
#         X = np.dot(rho, U.T)

#         e_t = np.zeros((2, 181))
#         for theta in range(181):
#             # rad = pi * deg / 180.0
#             e_t[0][theta] = math.cos(theta * math.pi/180.0)
#             e_t[1][theta] = math.sin(theta * math.pi/180.0)

#         # x_1 cos + x_2 sin
#         Y = np.dot(X, e_t)


#         nel = img.shape[0] * img.shape[1]
#         bin_width = np.zeros(181)
#         for theta in range(181):
#             bin_width[theta] = (3.5 * np.std(Y[:, :, theta])) * (nel ** (-1.0/3))

#         entropy = []
#         for theta in range(181):
#             temp = []
#             mean = np.mean(Y[:, :, theta])
#             std = np.std(Y[:, :, theta])

#             comp1 = mean-3*std
#             comp2 = mean+3*std
#             # CI 99.7%
#             for j in range(Y.shape[0]):
#                 for k in range(Y.shape[1]):
#                     if Y[j][k][theta] > comp1 and Y[j][k][theta] < comp2:
#                         temp.append(Y[j][k][theta])
#             nbins = round((max(temp)-min(temp))/bin_width[theta])
#             #  cal prob
#             hist, bins = np.histogram(temp, bins=nbins)
#             hist = filter(lambda var1: var1 != 0, hist)
#             hist1 = np.array([float(var) for var in hist])
#             hist1 = hist1/sum(hist1)

#             entropy.append(-1*sum(np.multiply(hist1, np.log2(hist1))))

#         min_angle = entropy.index(min(entropy))

#         e_t = np.array([math.cos(min_angle * math.pi/180.0),
#                     math.sin(min_angle * math.pi/180.0)])
       

#         # project to gray
#         # exp([x1 * cos, x2 * sin])
#         I1D = np.exp(np.dot(X, e_t))

#         # cos^2 + sin^2
#         p_theta = np.dot(e_t.T, e_t)
#         X_theta = X * p_theta

#         rho_ti = np.dot(X_theta, U)
#         c_ti = np.exp(rho_ti)
#         sum_ti = np.sum(c_ti, axis=2)
#         sum_ti = sum_ti.reshape(c_ti.shape[0], c_ti.shape[1], 1)
#         r_ti = c_ti/sum_ti
#         r_ti2 = 255 * r_ti
        
#         cv2.imwrite(f'/home/haoyu/Desktop/partical/ShadowNet_Data/test/test_intr2d_light/{img_name}.png',r_ti2) #path to directory where image is saved
#         # concat_img = np.concatenate((orig_img, r_ti2))
#         # cv2.imwrite(f'/home/haoyu/Desktop/partical/ShadowNet_Data/train_shadow_free_concat/{img_name}.png',concat_img) #path to directory where image is saved
        

#     except:
#         print(f"{img_name}")
        
# if __name__=='__main__':
#     path = "/home/haoyu/Desktop/partical/ShadowNet_Data/test/test_shadow_free"
#     files = sorted([os.path.join(path, x) for x in os.listdir(path) if (x.endswith(".jpg") or x.endswith(".png"))])
#     with Pool(16) as mp_pool:
#         mp_pool.map(illuminant, files)
        
        
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import math
import copy
from multiprocessing import Pool


def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)


def illuminant(img_path):
    img_name = img_path.split("/")[-1].split(".")[0]

    try:
        img = cv2.imread(img_path)
        img = np.float64(img)
        orig_img = copy.deepcopy(img)
        blue, green, red = cv2.split(img)

        # fix blue, green, red == 0
        blue[blue == 0] = 1
        green[green == 0] = 1
        red[red == 0] = 1


        # cal geometric mean of b、g、r => ck
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
        
        # r_ti = 255.0 * r_ti
        b_ti2 = 1 - r_ti[:, :, 0]
        b_ti2 =255.0 * b_ti2
        
        
        # r_ti2 = RGB2BGR(r_ti2)
        
        cv2.imwrite(f'/home/haoyu/Desktop/partical/ShadowNet_Data/train3/train_intr2d_light/{img_name}.png',b_ti2) #path to directory where image is saved
        # concat_img = np.concatenate((orig_img, r_ti2), axis=1)
        # cv2.imwrite(f'/home/haoyu/Desktop/partical/ShadowNet_Data/concat/{img_name}.png',concat_img) #path to directory where image is saved
        

    except:
        print(f"{img_name}")
        
if __name__=='__main__':
    path = "/home/haoyu/Desktop/partical/ShadowNet_Data/train3/train_shadow"
    files = sorted([os.path.join(path, x) for x in os.listdir(path) if (x.endswith(".jpg") or x.endswith(".png"))])
    with Pool(10) as mp_pool:
        mp_pool.map(illuminant, files)