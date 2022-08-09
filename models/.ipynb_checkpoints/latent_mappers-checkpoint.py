import torch
from torch import nn
from torch.nn import Module

from models.stylegan2.model import EqualLinear, PixelNorm
from torch.nn import Linear, LayerNorm, LeakyReLU, Sequential

class ModulationModule(Module):
    def __init__(self, layernum):
        super(ModulationModule, self).__init__()
        self.layernum = layernum
        self.fc = Linear(512, 512)
        self.norm = LayerNorm([self.layernum, 512], elementwise_affine=False)
        self.gamma_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, 512))
        self.beta_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, 512))
        self.leakyrelu = LeakyReLU()

    def forward(self, x, embedding):
        x = self.fc(x)
        x = self.norm(x)
        gamma = self.gamma_function(embedding.float())
        beta = self.beta_function(embedding.float())
        out = x * (1 + gamma) + beta
        out = self.leakyrelu(out)
        return out

class SubMapper(Module):
    def __init__(self, opts, layernum):
        super(SubMapper, self).__init__()
        self.opts = opts
        self.layernum = layernum
        self.pixelnorm = PixelNorm()
        self.modulation_module_list = nn.ModuleList([ModulationModule(self.layernum) for i in range(1)])

    def forward(self, x, embedding):
        x = self.pixelnorm(x)
        for modulation_module in self.modulation_module_list:
        	x = modulation_module(x, embedding)        
        return x

class LatentMapper(Module): 
    def __init__(self, opts):
        super(LatentMapper, self).__init__()
        self.opts = opts
        self.course_mapping = SubMapper(opts, 4)
        self.medium_mapping = SubMapper(opts, 4)
        self.fine_mapping = SubMapper(opts, 10)


    def forward(self, x, txt_embed):
        txt_embed = txt_embed.unsqueeze(1).repeat(1, 18, 1).detach()

        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:, :]

        x_coarse = self.course_mapping(x_coarse, txt_embed[:, :4, :])
        x_medium = self.medium_mapping(x_medium, txt_embed[:, 4:8, :])
        x_fine = self.fine_mapping(x_fine, txt_embed[:, 8:, :])
        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)
        return out