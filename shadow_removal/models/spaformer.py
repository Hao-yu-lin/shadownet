import torch
from torch import nn
import torch.nn.functional as F
from models.package.utils import weights_init, print_network
# import common
from models.package.transformer import *





class Generator(nn.Module):
    def __init__(self, input_nc, 
                 output_nc, 
                 dim=32, 
                 ffn_expansion_factor = 2.66,
                 LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
                 ):
        super(Generator, self).__init__()
        
        self.patch_embed = OverlapPatchEmbed(input_nc, dim)
        self.transformer_encoder = TransformerBlock(dim=dim, num_heads=1, ffn_expansion_factor=ffn_expansion_factor, bias=False, LayerNorm_type=LayerNorm_type)
        
    def forward(self, x):
        
        inp_embed = self.patch_embed(x)
        out = self.transformer_encoder(inp_embed)
        