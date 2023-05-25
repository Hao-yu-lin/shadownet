import numpy as np
import math
import torch
import torch.nn as nn
from PIL import Image
from timm.models.layers import to_2tuple
 

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=64, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        

        self.proj = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        )
    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class GlobalFilter(nn.Module):
    def __init__(self, dim, h=4, w=4):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x, spatial_size=None):

        B, N, C = x.shape
        if spatial_size is None:
            a = b = int(math.sqrt(N))
        else:
            a, b = spatial_size
        
        x = x.view(B, a, b, C)

        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        
        
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm='ortho')

        x = x.reshape(B, N, C)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, h, w , mlp_ratio=4.):
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.GlobalFilter = GlobalFilter(dim, h, w)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
        
    def forward(self, x):
        x = x + self.GlobalFilter(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        
        return x
    
class Classifier(nn.Module):
    def __init__(self, img_size=128,patch_size = [4, 8, 16], embed_dim = [32, 64, 128], n_blocks=4):
        super(Classifier, self).__init__()
        
        # B x C x H x W => B x patch x embed_dim
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=patch_size[0], in_chans=3, embed_dim=embed_dim[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size, patch_size=patch_size[1], in_chans=3, embed_dim=embed_dim[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size, patch_size=patch_size[2], in_chans=3, embed_dim=embed_dim[2])
        
        h = img_size // patch_size[0]
        w = h // 2 + 1
        
        self.block1 = nn.Sequential(*[TransformerBlock(dim=embed_dim[0], h=h, w=w) for _ in range(n_blocks)])
        self.norm1 = nn.LayerNorm(embed_dim[0])
        self.head1 = nn.Linear(embed_dim[0], 1)
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        x = self.patch_embed1(x)
        x = self.block1(x)
        x = self.norm1(x).mean(1)
        x = self.sigmoid(self.head1(x))
        
        out = torch.squeeze(x, 1)
        return out
    


