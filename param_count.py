import torch
from torch import nn
from torch.nn import Module
import sys

from models.stylegan2.model import PixelNorm
from torch.nn import Linear, LayerNorm, LeakyReLU, Sequential, InstanceNorm2d, Conv2d
import numpy as np

class TextModulationModule(Module):
    def __init__(self, in_channels):
        super(TextModulationModule, self).__init__()
        self.conv = Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False)
        self.norm = InstanceNorm2d(in_channels)
        # self.mapping = Sequential(Linear(512, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(), Linear(512, 512), LeakyReLU(), Linear(512, 512))
        self.gamma_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, in_channels))
        self.beta_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, in_channels))
        self.leakyrelu = LeakyReLU()
        
    def forward(self, x, embedding):
        x = self.conv(x)
        x = self.norm(x)
        embedding = self.mapping(embedding)
        log_gamma = self.gamma_function(embedding.float())
        gamma = log_gamma.exp().unsqueeze(2).unsqueeze(3)
        beta = self.beta_function(embedding.float()).unsqueeze(2).unsqueeze(3)
        out = x * (1 + gamma) + beta
        out = self.leakyrelu(out)
        return out
        
class SubTextMapper(Module):
    def __init__(self, in_channels):
        super(SubTextMapper, self).__init__()
        self.pixelnorm = PixelNorm()
        self.modulation_module_list = nn.ModuleList([TextModulationModule(in_channels) for i in range(1)])
        
    def forward(self, x, embedding):
        x = self.pixelnorm(x)
        for modulation_module in self.modulation_module_list:
            x = modulation_module(x, embedding)
        return x

class FeatureMapper(Module): 
    def __init__(self):
        super(FeatureMapper, self).__init__()
        self.mapping = SubTextMapper(512)

    def forward(self, features, txt_embed):
        txt_embed = txt_embed.detach()
        return self.mapping(features, txt_embed)
    
model = FeatureMapper()
total_params = sum(p.numel() for p in model.parameters())
print(total_params)