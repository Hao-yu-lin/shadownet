import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .package.library import *
# from package.library import *



# ngf：number of filters in the generator
# ndf：number of filters in the discriminator
class Generator(nn.Module):
    def __init__(self, input_c = 3,
                output_c = 3,
                n_blocks = [4, 6, 6, 8],
                img_size = 128,
                heads = [1,2,4,8],
                embed_dims=[32, 64, 128, 256],
                sr_ratios = [8, 4, 2, 1]):
        super(Generator, self).__init__()
        self.input_c = input_c
        self.output_c = output_c
        self.n_blocks = n_blocks
        self.len_blocks = len(n_blocks)
        self.sr_ratios = sr_ratios
        
        # patch_embed
        self.patch_embed_1 = OverlapPatchEmbed(img_size=img_size, patch_size=3, stride=2, in_chans=input_c,
                                              embed_dim=embed_dims[0])
        self.patch_embed_2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed_3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed_4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])
        
        for i in range(self.len_blocks):
            setattr(self, 'transformblock_' + str(i+1), nn.Sequential(*[TransformerBlock(dim = embed_dims[i], num_heads=heads[i], sr_ratio=self.sr_ratios[i]) for j in range(self.n_blocks[i])]))
        
        for i in range(self.len_blocks):
            setattr(self, 'resnetblock_' + str(i+1), nn.Sequential(*[ResnetBlock(dim=embed_dims[i]) for j in range(self.n_blocks[i])]))

        for i in range(self.len_blocks):
            setattr(self, 'GAPandGMP_' + str(i+1), GAPAndGMP(dim=embed_dims[i]))
        
        for i in range(self.len_blocks):
            setattr(self, 'GammaBeta_' + str(i+1), GammaBeta(dim=embed_dims[i]))
            
        for i in range(self.len_blocks):
            setattr(self, 'resnetadainblock_' + str(i+1), ResnetAdaILNBlock(dim=embed_dims[i]))

        for i in range(self.len_blocks):
            setattr(self, 'upsampling_' + str(i+1), UpSampling(dim=embed_dims[i], state="bilinear"))    # 'linear', 'bilinear', 'bicubic', or 'trilinear'.
            
        self.output_layer = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(embed_dims[0]//2, output_c, kernel_size=7, stride=1, padding=0, bias=False)
        )
    def forward(self, x):
        output1_ = self.patch_embed_1(x)
        output1_ = self.transformblock_1(output1_)
        
        output2_ = self.patch_embed_2(output1_)
        output2_ = self.transformblock_2(output2_)
        
        output3_ = self.patch_embed_3(output2_)
        output3_ = self.transformblock_3(output3_)
        
        output4_ = self.patch_embed_4(output3_)
        output4_ = self.transformblock_4(output4_)
        
        # B x embed_dims[3] x H/16 x W/16
        output4 = self.resnetblock_4(output4_)
        output4, _, _ = self.GAPandGMP_4(output4)
        gamma, beta = self.GammaBeta_4(output4)
        for i in range(self.n_blocks[3]):
            output4 = self.resnetadainblock_4(output4, gamma, beta)
        output4 = output4 + output4_ 
        output4 = self.upsampling_4(output4)    #  B x embed_dims[2] x H/8 x W/8
        
        # output3 = output3 + output4
        # B x embed_dims[2] x H/8 x W/8
        output3 = self.resnetblock_3(output4)
        output3, _, _ = self.GAPandGMP_3(output3)
        gamma, beta = self.GammaBeta_3(output3)
        for i in range(self.n_blocks[2]):
            output3 = self.resnetadainblock_3(output3, gamma, beta)
        output3 = output3 + output3_
        output3 = self.upsampling_3(output3)    # B x embed_dims[1] x H/4 x W/4
        
        # output2 = output2 + output3
        # B x embed_dims[1] x H/4 x W/4
        output2 = self.resnetblock_2(output3)
        output2, _, _ = self.GAPandGMP_2(output2)
        gamma, beta = self.GammaBeta_2(output2)
        for i in range(self.n_blocks[1]):
            output2 = self.resnetadainblock_2(output2, gamma, beta)
        output2 = output2 + output2_
        output2 = self.upsampling_2(output2)    # B x embed_dims[0] x H/2 x W/2
        
        # output1 = output2 + output1
        # B x embed_dims[0] x H/2 x W/2
        output1 = self.resnetblock_1(output2)
        output1, cam_logit, heat_map = self.GAPandGMP_1(output1)
        gamma, beta = self.GammaBeta_1(output1)
        for i in range(self.n_blocks[0]):
            output1 = self.resnetadainblock_1(output1, gamma, beta)
        output1 = output1 + output1_
        output1 = self.upsampling_1(output1)

        output = (self.output_layer(output1) + x).tanh()
        
        return output, cam_logit, heat_map
        

class Discriminator(nn.Module):
    def __init__(self, input_c=3, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        self.n_layers = n_layers
        self.initial = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(input_c, ndf, kernel_size=4, stride=2, padding=1, bias=False)),
            # nn.LeakyReLU(0.2, True)
            nn.ReLU(True),
        )
        
        for i in range(n_layers - 1):
            mult = 2**i
            setattr(self, 'disblock_' + str(i+1), DisBlock(ndf * mult, ndf * mult * 2))
        
        mult = 2 ** (n_layers - 1)
        self.gapandgmp = GAPAndGMP(ndf*mult)
        self.output = nn.Conv2d(ndf*mult, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
    def forward(self, x):
        x = self.initial(x)
        for i in range(self.n_layers-1):
            x = getattr(self, 'disblock_'+ str(i+1))(x)
            # print(x.shape)
        x, cam_logit, heatmap = self.gapandgmp(x)
        out = self.output(x)
        # print(out.shape)
        return out, cam_logit, heatmap
    

class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w
            
        
if __name__ == '__main__':
    img = torch.rand(1, 3, 128, 128)
    # gen = Generator()
    dis = Discriminator(n_layers=5)
    dis(img)
    # gap_and_gmp = GAPAndGMP(32)
    
    # gap_and_gmp(img)
 

        