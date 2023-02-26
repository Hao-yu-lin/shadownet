'''
Restormer:https://github.com/swz30/Restormer
SegFormer:https://github.com/NVlabs/SegFormer
LG-ShadowNet:https://github.com/hhqweasd/LG-ShadowNet
G2R-ShadowNet:https://github.com/hhqweasd/G2R-ShadowNet
DC-SHadowNet:https://github.com/jinyeying/DC-ShadowNet-Hard-and-Soft-Shadow-Removal
SpA-Former:https://github.com/zhangbaijin/SpA-Former-shadow-removal
'''


import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from collections import OrderedDict
from .utils import *
from einops import rearrange

class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                    nn.InstanceNorm2d(dim),
                    nn.ReLU(True),
                    nn.ReflectionPad2d(1),
                    nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                    nn.InstanceNorm2d(dim) 
        )
        
    def forward(self, x):
        out = x + self.conv_block(x)
        return out
    
class ResnetAdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias=False):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        
        self.layernorm = nn.LayerNorm(dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.layernorm(x)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x



class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=32):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size 
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
        #                       padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.proj = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=2, padding=0),
            nn.InstanceNorm2d(embed_dim),
            nn.ReLU(True)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(init_weights)


    def forward(self, x):
        x = self.proj(x)
        
        # B, C, H, W = x.shape
        # x = x.flatten(2).transpose(1, 2)    # B, H*W, C
        # # x = self.norm(x)
        
        # x = x.transpose(1, 2).view(B, C, H, W)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
        
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, bias=False):
        super(TransformerBlock, self).__init__()
        
        # self.norm1 = LayerNorm(dim)
        self.norm1 = nn.InstanceNorm2d(dim)
        self.attn = Attention(
            dim, 
            num_heads=num_heads,
            bias=bias,
        )
        # self.norm2 = LayerNorm(dim)
        self.norm2 = nn.InstanceNorm2d(dim)
        self.ffn = FeedForward(dim)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# Global Average Pooling and Global Max Pooling
class GAPAndGMP(nn.Module):
    def __init__(self, dim):
        super(GAPAndGMP, self).__init__()
        
        self.gap_fc = nn.Linear(dim, 1, bias=False)
        self.gmp_fc = nn.Linear(dim, 1, bias=False)
        self.conv1x1 = nn.Conv2d(dim*2, dim, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)
        
        
    def forward(self, x):
        gap = torch.nn.functional.adaptive_avg_pool2d(x,1)      # B x C x 1 x 1
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))       # B x 1
        gap_weight = list(self.gap_fc.parameters())[0]          # B x C
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)          # B x C x H x W
        
        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        
        cam_logit = torch.cat([gap_logit, gmp_logit], 1)       # B x 2    

        x = torch.cat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))
        heatmap = torch.sum(x, dim=1, keepdim=True)         # B x 1 x H x W
        return x, cam_logit, heatmap
        
#  Gamma, Beta block
class GammaBeta(nn.Module):
    def __init__(self, dim):
        super(GammaBeta, self).__init__()
        
        self.FC = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(True),
            nn.Linear(dim, dim, bias=False),
            nn.ReLU(True)
        )
        
        self.gamma = nn.Linear(dim, dim, bias=False)
        self.beta = nn.Linear(dim, dim, bias=False)
        
    def forward(self, x):
        x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        x_ = self.FC(x_.view(x_.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_)
        
        return gamma, beta        
        
         
class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out
    
class UpSampling(nn.Module):
    def __init__(self, dim, state):
        super(UpSampling, self).__init__()
        self.UpBlock = nn.Sequential(
            nn.Upsample(scale_factor=2, mode=state),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, int(dim / 2), kernel_size=3, stride=1, padding=0, bias=False),
            ILN(int(dim / 2)),
            nn.ReLU(True)
        )
        
    def forward(self, input):
        output = self.UpBlock(input)
        return output
    
    
class DisBlock(nn.Module):
    def __init__(self, input_c, output_c):
        super(DisBlock, self).__init__()
        
        self.fc = nn.utils.spectral_norm(
            nn.Conv2d(input_c, output_c, kernel_size=4, stride=2, padding=1, bias=False)
        )
        
        self.activation = nn.ReLU(True)
        
    def forward(self, input):
        output = self.activation(self.fc(input))

        return output