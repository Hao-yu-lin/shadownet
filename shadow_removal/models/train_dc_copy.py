from models.dcshadownet import *
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageFolder
from models.illuminant import illuminant
from tqdm.contrib import tzip
from tqdm.auto import tqdm
from multiprocessing import Pool
from datetime import datetime
from pytorch_msssim import ms_ssim, ssim
import numpy as np
import os
import cv2
import itertools
from torchsummary import summary

def denorm(x):
    return x * 0.5 + 0.5

def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

def avg(x):
    return sum(x)/len(x)


class DCShadowNet(object):
    def __init__(self, args):
        
        self.train_path = args.train_path
        self.test_path = args.test_path
        self.iteration = args.iteration
        
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch
        
        self.G_update = args.g_update
        self.D_update = args.d_update

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.dom_weight = args.dom_weight
        exp_time = datetime.today().strftime("%m-%d-%H")
        
        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.output_folder = os.path.join(args.result_dir, exp_time)
        self.ckpt_folder = os.path.join(args.ckpt_dir, exp_time)
        os.makedirs(self.output_folder, exist_ok = True)
        os.makedirs(self.ckpt_folder, exist_ok = True)
        
        
    def __img_loader(self, path, tfm=None, state="train"):
        state = {
            "shadow" : (f"{state}_shadow", f"{state}_intr2d_light"),
            "shadow_free" : (f"{state}_shadow_free", f"{state}_intr2d_light"),
        }
        
        sh_img = ImageFolder(path, state["shadow"], tfm)
        sh_free_img = ImageFolder(path, state["shadow_free"], tfm)

        return sh_img, sh_free_img
    
    def build_model(self):
        tfm = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
            
        ])
        
        self.train_sh_img,  self.train_sh_free_img = self.__img_loader(self.train_path, tfm, state="train")
        self.train_loader = DataLoader(self.train_sh_img, batch_size=self.batch_size, shuffle=True)
        self.train_sh_free_loader = DataLoader(self.train_sh_free_img, batch_size=self.batch_size, shuffle=True)
        
        
        self.test_sh_img,  self.test_sh_free_img = self.__img_loader(self.test_path, state="test")
        self.test_loader = DataLoader(self.test_sh_img, batch_size=1, shuffle=False)
        
        
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=True).to(self.device)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=True).to(self.device)
        #  use to detect shadow is real or fake
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        #  use to detect shadow_free is real or fake
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        # self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
        # self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
        
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)
        

        """ Trainer """
        self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.Rho_clipper = RhoClipper(0, 1)
        
        
    def train(self):
        summary(self.genA2B, (3, 256, 256))
        milestones = [x for x in range(8000) if x % 1000 == 0 and x != 0]
        self.G_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.G_optim, milestones=milestones)
        self.D_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.D_optim, milestones=milestones)
        
        G_ad_A = []
        G_ad_Dom_A = []
        G_recon_A = []
        G_identity_A = []
        G_dom_A = []
        
        G_ad_B = []
        G_ad_Dom_B = []
        G_recon_B = []
        G_identity_B = []
        G_dom_B = []
        
        ch = []
        
        for step in range(self.iteration):
            
            self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train()
            
            try:
                shadow, shadow_intr2d = train_shadow.next()
            except:
                train_shadow = iter(self.train_loader)
                shadow, shadow_intr2d = train_shadow.next()
            
            try:
                sh_free, sh_free_intr2d = train_shadow_free.next()
            except:
                train_shadow_free = iter(self.train_sh_free_loader)
                sh_free, sh_free_intr2d = train_shadow.next()
           
           
            shadow = shadow.to(self.device)
            
            sh_free = sh_free.to(self.device)
            
            # Update D
            # A:shadow B:shadow_free
            if step % self.D_update == 0:
                self.D_optim.zero_grad()
            fake_A2B, _, _ = self.genA2B(shadow)
            fake_B2A, _, _ = self.genB2A(sh_free)
            
            real_GA_logit, real_GA_Dom_logit, _ = self.disGA(shadow)
            real_GB_logit, real_GB_Dom_logit, _ = self.disGB(sh_free)
            
            fake_GA_logit, fake_GA_Dom_logit, _ = self.disGA(fake_B2A)
            fake_GB_logit, fake_GB_Dom_logit, _ = self.disGB(fake_A2B)
            
            D_ad_loss_GA     = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit,  torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_Dom_loss_GA = self.MSE_loss(real_GA_Dom_logit, torch.ones_like(real_GA_Dom_logit).to(self.device)) + self.MSE_loss(fake_GA_Dom_logit, torch.zeros_like(fake_GA_Dom_logit).to(self.device))
            
            D_ad_loss_GB     = self.MSE_loss(real_GB_logit,     torch.ones_like(real_GB_logit).to(self.device))     + self.MSE_loss(fake_GB_logit,     torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_Dom_loss_GB = self.MSE_loss(real_GB_Dom_logit, torch.ones_like(real_GB_Dom_logit).to(self.device)) + self.MSE_loss(fake_GB_Dom_logit, torch.zeros_like(fake_GB_Dom_logit).to(self.device))
            
            D_loss_A = D_ad_loss_GA + D_ad_Dom_loss_GA
            D_loss_B = D_ad_loss_GB + D_ad_Dom_loss_GB
            
            D_loss = D_loss_A + D_loss_B
            
            D_loss.backward()
            
            if step % self.D_update == 0:    
                self.D_optim.step()
            
            
            # Update G
            if step % self.G_update == 0:
                self.G_optim.zero_grad()
            # A:shadow B:shadow_free
            
            fake_A2B, fake_A2B_Dom_logit, _ = self.genA2B(shadow)
            fake_B2A, fake_B2A_Dom_logit, _ = self.genB2A(sh_free)
            
            fake_A2B2A, _, _ = self.genB2A(fake_A2B)    
            fake_B2A2B, _, _ = self.genA2B(fake_B2A)    
            
            fake_A2A, fake_A2A_Dom_logit, _ = self.genB2A(shadow)   
            fake_B2B, fake_B2B_Dom_logit, _ = self.genA2B(sh_free)  
            
            fake_GA_logit, fake_GA_Dom_logit, _ = self.disGA(fake_B2A)  
            fake_GB_logit, fake_GB_Dom_logit, _ = self.disGB(fake_A2B)  

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_Dom_loss_GA = self.MSE_loss(fake_GA_Dom_logit, torch.ones_like(fake_GA_Dom_logit).to(self.device))           
            
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))             
            G_ad_Dom_loss_GB = self.MSE_loss(fake_GB_Dom_logit, torch.ones_like(fake_GB_Dom_logit).to(self.device))
            
            G_recon_loss_A = self.L1_loss(fake_A2B2A, shadow)   
            G_recon_loss_B = self.L1_loss(fake_B2A2B, sh_free)
            
            G_identity_loss_A = self.L1_loss(fake_A2A, shadow)   # 確保圖片輸出一致性
            G_identity_loss_B = self.L1_loss(fake_B2B, sh_free)

            G_dom_loss_A = self.BCE_loss(fake_B2A_Dom_logit, torch.ones_like(fake_B2A_Dom_logit).to(self.device)) + self.BCE_loss(fake_A2A_Dom_logit, torch.zeros_like(fake_A2A_Dom_logit).to(self.device)) ##fake_A, 1; same_A(fake_A2A) 0
            G_dom_loss_B = self.BCE_loss(fake_A2B_Dom_logit, torch.ones_like(fake_A2B_Dom_logit).to(self.device)) + self.BCE_loss(fake_B2B_Dom_logit, torch.zeros_like(fake_B2B_Dom_logit).to(self.device)) ##fake_B, 1; same_B(fake_B2B) 0

            ill_sh = illuminant(fake_A2B)
            ill_sh = ill_sh.to(self.device)
            
            ill_shf = illuminant(fake_B2A)
            ill_shf = ill_shf.to(self.device)
            shadow_intr2d = shadow_intr2d.to(self.device)
            sh_free_intr2d = sh_free_intr2d.to(self.device)
            # print(shadow_intr2d)
            # print("---------------")
            # print(ill_sh)
            
            loss_ch = self.L1_loss(shadow_intr2d, ill_sh) + self.L1_loss(sh_free_intr2d, ill_shf)
            # loss_ch = 1 - ms_ssim(shadow_intr2d, ill_sh)
            
            
            # G_loss_A = G_ad_loss_GA + G_ad_Dom_loss_GA + G_recon_loss_A + G_identity_loss_A + G_dom_loss_A
            # G_loss_B = G_ad_loss_GB + G_ad_Dom_loss_GB + G_recon_loss_B + G_identity_loss_B + G_dom_loss_B 
            
            # G_loss = G_loss_A + G_loss_B + loss_ch
            G_ad_loss = G_ad_loss_GA + G_ad_loss_GB + G_ad_Dom_loss_GA + G_ad_Dom_loss_GB
            G_recon_loss = G_recon_loss_A + G_recon_loss_B
            G_identity_loss = G_identity_loss_A + G_identity_loss_B
            G_dom_loss = G_dom_loss_A + G_dom_loss_B
            
            # G_loss = G_ad_loss * 2 + G_recon_loss * 15  + G_dom_loss + loss_ch * 5
            G_loss = G_ad_loss * 2 + G_recon_loss * 15 + G_identity_loss * 15 + G_dom_loss +loss_ch * 5
            
            # G_loss = G_ad_loss * 2 + G_identity_loss * 10 + G_dom_loss + loss_ch * 5
            
            # G_loss = G_ad_loss + G_recon_loss * 10 + G_identity_loss * 10 + loss_ch * 5
            
            
            G_ad_A.append(G_ad_loss_GA.item())
            G_ad_Dom_A.append(G_ad_Dom_loss_GA.item())
            G_recon_A.append(G_recon_loss_A.item())
            G_identity_A.append(G_identity_loss_A.item())
            G_dom_A.append(G_dom_loss_A.item())
            
            G_ad_B.append(G_ad_loss_GB.item())
            G_ad_Dom_B.append(G_ad_Dom_loss_GB.item())
            G_recon_B.append(G_recon_loss_B.item())
            G_identity_B.append(G_identity_loss_B.item())
            G_dom_B.append(G_dom_loss_B.item())
            
            ch.append(loss_ch.item())
            
            G_loss.backward()
            
            if step % self.G_update == 0:
                self.G_optim.step()
            
        
            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)
            
            print("[%5d/%5d] d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, D_loss, G_loss))
            
            if (step % self.print_freq == 0) or (step == self.iteration-1):
                self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval()
                
                for idx, batch in enumerate(tqdm(self.test_loader)):
                    imgs, _ = batch
                    with torch.no_grad():
                        outputs, _, _ = self.genA2B(imgs.to(self.device))
                    
                    # outputs = torch.squeeze(outputs, dim = 0)
                    
                    concat_img = np.concatenate((RGB2BGR(tensor2numpy(denorm(imgs[0]))), RGB2BGR(tensor2numpy(denorm(outputs[0])))))
                    
                    cv2.imwrite(os.path.join(self.output_folder, f"{idx}_{step}.png"), concat_img * 255.0)
                # torch.save(self.gen.state_dict(), os.path.join(self.ckpt_folder, f"{step}_gen.ckpt"))
                # torch.save(self.dis.state_dict(), os.path.join(self.ckpt_folder, f"{step}_dis.ckpt"))
                
                # params = {}
                # params['genA2B'] = self.genA2B.state_dict()
                # params['genB2A'] = self.genB2A.state_dict()
                # params['disGA'] = self.disGA.state_dict()
                # params['disGB'] = self.disGB.state_dict()
                
                # torch.save(params, os.path.join(self.ckpt_folder, f'_params_{step}.pt'))
                
                G_ad_A = avg(G_ad_A)
                G_ad_Dom_A = avg(G_ad_Dom_A)
                G_recon_A = avg(G_recon_A)
                G_identity_A = avg(G_identity_A)
                G_dom_A = avg(G_dom_A)
                
                G_ad_B = avg(G_ad_B)
                G_ad_Dom_B = avg(G_ad_Dom_B)
                G_recon_B = avg(G_recon_B)
                G_identity_B = avg(G_identity_B) 
                G_dom_B = avg(G_dom_B)
                
                ch = avg(ch)
                
                avg_G_A_loss = G_ad_A + G_ad_Dom_A + G_recon_A + G_identity_A + G_dom_A
                avg_G_B_loss = G_ad_B + G_ad_Dom_B + G_recon_B + G_identity_B + G_dom_B
                
                avg_G_loss = avg_G_A_loss + avg_G_B_loss
                
                with open(os.path.join(self.ckpt_folder, "loss.txt"), "a") as the_file:
                    the_file.write(
                        "[%5d/%5d] G_ad_loss_GA: %.8f, G_ad_Dom_loss_GA: %.8f, G_recon_loss_A: %.8f, G_identity_loss_A: %.8f, G_dom_loss_A: %.8f\n" 
                            % (step, self.iteration, G_ad_A, G_ad_Dom_A, G_recon_A, G_identity_A, G_dom_A) # , G_dom_loss_A: %.8f
                    )
                    the_file.write(
                        "[%5d/%5d] G_ad_loss_GB: %.8f, G_ad_Dom_loss_GB: %.8f, G_recon_loss_B: %.8f, G_identity_loss_B: %.8f, G_dom_loss_B: %.8f\n" 
                            % (step, self.iteration, G_ad_B, G_ad_Dom_B, G_recon_B, G_identity_B, G_dom_B)  # , G_dom_loss_B: %.8f
                    )
                    the_file.write(
                        "[%5d/%5d] ch_loss: %.8f, G_A_loss: %.8f, G_B_loss: %.8f, G_loss: %.8f \n" 
                            % (step, self.iteration, ch, avg_G_A_loss, avg_G_B_loss, avg_G_loss)
                    
                    )
                
                G_ad_A = []
                G_ad_Dom_A = []
                G_recon_A = []
                G_identity_A = []
                G_dom_A = []
                
                G_ad_B = []
                G_ad_Dom_B = []
                G_recon_B = []
                G_identity_B = []
                G_dom_B = []
                
                ch = []
                
            self.G_scheduler.step()
            self.D_scheduler.step()