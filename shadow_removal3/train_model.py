import os
import cv2
import itertools
import numpy as np
import torch
import torch.nn as nn

from models.network import Discriminator, Generator, RhoClipper
from models.class_module import Classifier
from models.illuminant import illuminant
# from models.dcshadownet import ResnetGenerator, RhoClipper
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageFolder
from tqdm.auto import tqdm
from datetime import datetime
from laplacian_pyramid_loss import LaplacianPyramidLoss




def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

def avg(x):
    return sum(x)/len(x)

class ShadowModel(object):
    def __init__(self, args):
        
        self.train_path = args.train_path
        self.test_path = args.test_path
        self.iteration = args.iteration
        self.pretrained = args.pre_trained
        self.pretrain_path = args.prepretrain_path
        
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        
        self.G_update = args.g_update
        self.D_update = args.d_update

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.dom_weight = args.dom_weight
        self.ill_weight = args.ill_weight
        
        self.norm_type = args.norm_type
        
        if(self.norm_type == 'norm1'):
            self.norm_mean = [0.485,0.456,0.406]
            self.norm_std = [0.229,0.224,0.225]
            self.denorm = self.__denorm1
        elif(self.norm_type == 'norm2'):
            self.norm_mean = 0.5
            self.norm_std = 0.5
            self.denorm = self.__denorm2
        else:
            raise RuntimeError('norm type erro! ')
        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis
        
        """ Other """
        exp_time = datetime.today().strftime("%m-%d-%H")
        self.device = args.device
        self.output_folder = os.path.join(args.result_dir, exp_time)
        self.ckpt_folder = os.path.join(args.ckpt_dir, exp_time)
        os.makedirs(self.output_folder, exist_ok = True)
        os.makedirs(self.ckpt_folder, exist_ok = True)
        
        self.img_size = args.img_size
        
    def __denorm1(self, x):
        mean=[0.485,0.456,0.406]
        std=[0.229,0.224,0.225]
        for i in range(3):
            x[i] = x[i] * std[i] + mean[i]
        # return x * 0.5 + 0.5
        return x

    def __denorm2(self, x):
        return x * 0.5 + 0.5
        
    def __img_loader(self, path, tfm=None, state="train"):
        state = {
            "shadow" : (f"{state}_shadow", f"{state}_intr2d_light"),
            "shadow_free" : (f"{state}_shadow_free", f"{state}_intr2d_light"),
        }
        
        sh_img = ImageFolder(path, state["shadow"], tfm)
        sh_free_img = ImageFolder(path, state["shadow_free"], tfm)

        return sh_img, sh_free_img
    
    def build_model(self):
        train_tfm = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
        ])
        
        self.train_sh_img,  self.train_sh_free_img = self.__img_loader(self.train_path, train_tfm, state="train")
        self.train_loader = DataLoader(self.train_sh_img, batch_size=self.batch_size, shuffle=True)
        self.train_sh_free_loader = DataLoader(self.train_sh_free_img, batch_size=self.batch_size, shuffle=True)
        
        test_tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
        ])
        self.test_sh_img,  self.test_sh_free_img = self.__img_loader(self.test_path, test_tfm, state="test")
        self.test_loader = DataLoader(self.test_sh_img, batch_size=1, shuffle=False)
        
        self.genA2B = Generator().to(self.device)
        self.genB2A = Generator().to(self.device)
        
        
        self.disGA = Discriminator().to(self.device)
        self.disGB = Discriminator().to(self.device)
        # self.disLA = Discriminator(ndf=128,n_layers=3).to(self.device)
        # self.disLB = Discriminator(ndf=128,n_layers=3).to(self.device)
        
        self.disCA = Classifier(self.img_size).to(self.device)
        self.disCB = Classifier(self.img_size).to(self.device)
        
        self.shadow_class = Classifier(self.img_size).to(self.device)
        
        ''' Loss Function'''
        self.L1_loss = nn.L1Loss().to(self.device)
        self.Laplace_loss = LaplacianPyramidLoss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCElog_loss = nn.BCEWithLogitsLoss().to(self.device)
        self.BCE_loss = nn.BCELoss(reduce='mean')
        
        
        ''' Optimizer '''
        self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        # self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        # self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters()), weight_decay=self.weight_decay)
        self.C_optim = torch.optim.SGD(itertools.chain(self.disCA.parameters(), self.disCB.parameters()), lr=self.lr, weight_decay=self.weight_decay) 
        
        
        self.Rho_clipper = RhoClipper(0, 1)
        
        # milestones = [x for x in range(8000) if x % 1000 == 0 and x != 0]
        # self.G_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.G_optim, milestones=milestones)
        # self.D_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.D_optim, milestones=milestones)
        
        
    def train(self):
        
        if(self.pretrained):
            param = torch.load(self.pretrain_path)
            self.genA2B.load_state_dict(param['genA2B'])
            self.genB2A.load_state_dict(param['genB2A'])
            self.disGA.load_state_dict(param['disGA'])
            self.disGB.load_state_dict(param['disGB'])
            # self.disLA.load_state_dict(param['disLA'])
            # self.disLB.load_state_dict(param['disLB'])
            
        self.shadow_class.load_state_dict(torch.load("/home/haoyu/Desktop/partical/shadow_removal3/output/ckpt/05-06-15/best.ckpt"))
        
        for step in range(self.iteration):
            self.genA2B.train(), self.genB2A.train()
            self.disGA.train(), self.disGB.train()
            self.disCA.train(), self.disCB.train()
            self.shadow_class.eval()
            
            try:
                shadow, shadow_intr2d = train_shadow.next()
            except:
                train_shadow = iter(self.train_loader)
                shadow, shadow_intr2d = train_shadow.next()
            
            try:
                sh_free, sh_free_intr2d = train_shadow_free.next()
            except:
                train_shadow_free = iter(self.train_sh_free_loader)
                sh_free, sh_free_intr2d = train_shadow_free.next()


            
            shadow = shadow.to(self.device)
            sh_free = sh_free.to(self.device)
            
            # Update D
            
            # A:shadow, B:shadow_free
            if step % self.D_update == 0:
                self.D_optim.zero_grad()
            
            fake_A2B, _, _ = self.genA2B(shadow)
            fake_B2A, _, _ = self.genB2A(sh_free)
            
            # disGA, disGB
            real_A_logit, real_A_Dom_logit, _ = self.disGA(shadow)
            real_B_logit, real_B_Dom_logit, _ = self.disGB(sh_free)
            
            fake_A_logit, fake_A_Dom_logit, _ = self.disGA(fake_B2A)
            fake_B_logit, fake_B_Dom_logit, _ = self.disGB(fake_A2B)
            
            D_ad_loss_A     = self.MSE_loss(real_A_logit, torch.ones_like(real_A_logit).to(self.device)) + self.MSE_loss(fake_A_logit, torch.zeros_like(fake_A_logit).to(self.device))
            D_ad_Dom_loss_A = self.MSE_loss(real_A_Dom_logit, torch.ones_like(real_A_Dom_logit).to(self.device)) + self.MSE_loss(fake_A_Dom_logit, torch.zeros_like(fake_A_Dom_logit).to(self.device))
            
            D_ad_loss_B     = self.MSE_loss(real_B_logit, torch.ones_like(real_B_logit).to(self.device)) + self.MSE_loss(fake_B_logit, torch.zeros_like(fake_B_logit).to(self.device))
            D_ad_Dom_loss_B = self.MSE_loss(real_B_Dom_logit, torch.ones_like(real_B_Dom_logit).to(self.device)) + self.MSE_loss(fake_B_Dom_logit, torch.zeros_like(fake_B_Dom_logit).to(self.device))
            
            D_loss_GA = D_ad_loss_A + D_ad_Dom_loss_A
            D_loss_GB = D_ad_loss_B + D_ad_Dom_loss_B
            
            real_A_logit = self.disCA(shadow)
            real_B_logit = self.disCB(sh_free)
            
            fake_A_logit = self.disCA(fake_B2A)
            fake_B_logit = self.disCB(fake_A2B)
            
            D_loss_A = self.BCE_loss(real_A_logit, torch.ones_like(real_A_logit).to(self.device)) + self.BCE_loss(fake_A_logit, torch.zeros_like(fake_A_logit).to(self.device))
            D_loss_B = self.BCE_loss(real_B_logit, torch.ones_like(real_B_logit).to(self.device)) + self.BCE_loss(fake_B_logit, torch.zeros_like(fake_B_logit).to(self.device))
            
            # D_loss update
            
            D_loss_G = D_loss_GA + D_loss_GB
            D_loss_C =  D_loss_A + D_loss_B
            D_loss = D_loss_G + D_loss_C
            
            
            
            D_loss.backward()
                        
            if step % self.D_update == 0:    
                self.D_optim.step()
            
            
            # Update G
            if step % self.G_update == 0:
                self.G_optim.zero_grad()
               
            # A:shadow B:shadow_free
            
            # fake_A2B, fake_A2B_Dom_logit, _ = self.genA2B(shadow)
            # fake_B2A, fake_B2A_Dom_logit, _ = self.genB2A(sh_free)
            fake_A2B, _, _ = self.genA2B(shadow)
            fake_B2A, _, _ = self.genB2A(sh_free)
            
            fake_A2B2A, _, _ = self.genB2A(fake_A2B)    
            fake_B2A2B, _, _ = self.genA2B(fake_B2A)   
            
            # fake_A2A, fake_A2A_Dom_logit, _ = self.genB2A(shadow)   
            # fake_B2B, fake_B2B_Dom_logit, _ = self.genA2B(sh_free)  
            fake_A2A, _, _ = self.genB2A(shadow)   
            fake_B2B, _, _ = self.genA2B(sh_free)  
            
            class_B = self.shadow_class(fake_A2B)
            class_A = self.shadow_class(fake_B2A)
            
            class_loss_B = self.BCE_loss(class_B, torch.zeros_like(class_B))
            class_loss_A = self.BCE_loss(class_A, torch.ones_like(class_A))
            
            class_loss = class_loss_A + class_loss_B
            
            # cal G loss
            
            G_recon_loss_A = self.Laplace_loss(fake_A2B2A, shadow)   
            G_recon_loss_B = self.Laplace_loss(fake_B2A2B, sh_free)
            
            G_identity_loss_A = self.Laplace_loss(fake_A2A, shadow)   # 確保圖片輸出一致性
            G_identity_loss_B = self.Laplace_loss(fake_B2B, sh_free)
            
        
            
            # G_dom_loss_A = self.BCElog_loss(fake_B2A_Dom_logit, torch.ones_like(fake_B2A_Dom_logit).to(self.device)) + self.BCElog_loss(fake_A2A_Dom_logit, torch.zeros_like(fake_A2A_Dom_logit).to(self.device)) ##fake_A, 1; same_A(fake_A2A) 0
            # G_dom_loss_B = self.BCElog_loss(fake_A2B_Dom_logit, torch.ones_like(fake_A2B_Dom_logit).to(self.device)) + self.BCElog_loss(fake_B2B_Dom_logit, torch.zeros_like(fake_B2B_Dom_logit).to(self.device)) ##fake_B, 1; same_B(fake_B2B) 0
            
            G_cycle_loss = self.cycle_weight * (G_recon_loss_A + G_recon_loss_B)
            G_identity_loss = self.identity_weight * (G_identity_loss_A + G_identity_loss_B)
            # G_dom_loss = self.dom_weight * (G_dom_loss_A + G_dom_loss_B)
            
            # cal D loss
            
            # fake_A_logit, fake_A_Dom_logit, _ = self.disGA(fake_B2A)  
            # fake_B_logit, fake_B_Dom_logit, _ = self.disGB(fake_A2B)  
            fake_A_logit, _, _ = self.disGA(fake_B2A)  
            fake_B_logit, _, _ = self.disGB(fake_A2B)  
            
            G_ad_loss_GA = self.MSE_loss(fake_A_logit, torch.ones_like(fake_A_logit).to(self.device))
            # G_ad_Dom_loss_GA = self.MSE_loss(fake_A_Dom_logit, torch.ones_like(fake_A_Dom_logit).to(self.device))           
            
            G_ad_loss_GB = self.MSE_loss(fake_B_logit, torch.ones_like(fake_B_logit).to(self.device))             
            # G_ad_Dom_loss_GB = self.MSE_loss(fake_B_Dom_logit, torch.ones_like(fake_B_Dom_logit).to(self.device))
            
            # loss_D_G = G_ad_loss_GA + G_ad_Dom_loss_GA + G_ad_loss_GB + G_ad_Dom_loss_GB
            loss_D_G = G_ad_loss_GA  + G_ad_loss_GB
        
            fake_A_logit = self.disCA(fake_A2A)  
            fake_B_logit = self.disCB(fake_B2B)  
            
            G_ad_loss_CA = self.BCE_loss(fake_A_logit, torch.ones_like(fake_A_logit).to(self.device))
            
            G_ad_loss_CB = self.BCE_loss(fake_B_logit, torch.ones_like(fake_B_logit).to(self.device))             
            
            loss_D_L = G_ad_loss_CA + G_ad_loss_CB
            
            loss_D = self.adv_weight * (loss_D_G + loss_D_L)
            
            # illuminate
            
            # ill_sh = illuminant(fake_A2B)
            # ill_sh = ill_sh.to(self.device)
            
            # ill_shf = illuminant(fake_B2A)
            # ill_shf = ill_shf.to(self.device)
            
            # shadow_intr2d = shadow_intr2d.to(self.device)
            # sh_free_intr2d = sh_free_intr2d.to(self.device)
            
            # loss_ch = self.L1_loss(shadow_intr2d, ill_sh) + self.L1_loss(sh_free_intr2d, ill_shf)
            # loss_ch = loss_ch *  self.ill_weight
            
            # G_loss = loss_D + G_cycle_loss + G_identity_loss + G_dom_loss + class_loss
            G_loss = loss_D + G_cycle_loss + G_identity_loss + class_loss
            
            
            G_loss.backward()
            if step % self.G_update == 0:
                self.G_optim.step()

            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)
            

            
            print("[%5d/%5d] d_loss: %.8f, g_loss: %.8f, class_loss: %.8f" % (step, self.iteration, D_loss, G_loss, class_loss))
            
            if (step % self.print_freq == 0) or (step == self.iteration-1):
                self.genA2B.eval(), self.genB2A.eval()
            
                for idx, batch in enumerate(tqdm(self.test_loader)):
                    imgs, _ = batch
                    with torch.no_grad():
                        outputs, _, _= self.genA2B(imgs.to(self.device))
                    
                    
                    # outputs3 = outputs[0] + heatmap
                    # outputs = torch.squeeze(outputs, dim = 0)
                    
                    concat_img = np.concatenate((RGB2BGR(tensor2numpy(self.denorm(imgs[0]))), RGB2BGR(tensor2numpy(self.denorm(outputs[0])))) ,axis = 1)
                    # concat_img = np.concatenate((concat_img, RGB2BGR(tensor2numpy(self.denorm(outputs3)))) ,axis = 1)

                    
                    cv2.imwrite(os.path.join(self.output_folder, f"{idx}_{step}.png"), concat_img * 255.0)
                params = {}
                params['genA2B'] = self.genA2B.state_dict()
                params['genB2A'] = self.genB2A.state_dict()
                params['disGA'] = self.disGA.state_dict()
                params['disGB'] = self.disGB.state_dict()
                params['disCA'] = self.disCA.state_dict()
                params['disCB'] = self.disCB.state_dict()
                
                
                torch.save(params, os.path.join(self.ckpt_folder, f'train.pt'))
                    
            
            # self.G_scheduler.step()
            # self.D_scheduler.step()