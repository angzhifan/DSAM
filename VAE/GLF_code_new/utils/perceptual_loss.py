from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
import torchvision.models as models


class _netVGGFeatures(nn.Module):
    def __init__(self,device):
        super(_netVGGFeatures, self).__init__()
        self.vggnet = models.vgg16(pretrained=True).to(device)
        self.layer_ids = [2, 7, 12, 21, 30]

    def main(self, z, levels):
        layer_ids = self.layer_ids[:levels]
        id_max = layer_ids[-1] + 1
        output = []
        for i in range(id_max):
            z = self.vggnet.features[i](z)
            if i in layer_ids:
                output.append(z)
        return output

    def forward(self, z, levels):
        output = self.main(z, levels)
        return output


class _VGGDistance(nn.Module):
    def __init__(self, levels, device):
        super(_VGGDistance, self).__init__()
        self.vgg = _netVGGFeatures(device)
        self.levels = levels
        self.factors = [0] * (self.levels + 1)
        self.pool = nn.AvgPool2d(8, 8)
        self.mse = nn.MSELoss(reduction='sum')
        
    def forward(self, I1, I2, use_factors=False):
        eps = 1e-8
        sum_factors = sum(self.factors)
        f1 = self.vgg(I1, self.levels)
        f2 = self.vgg(I2, self.levels)
        loss = torch.abs(I1 - I2).sum()
        for i in range(self.levels):
            layer_loss = torch.abs(f1[i] - f2[i]).sum()
            self.factors[i] += layer_loss.item()
            if use_factors:
                layer_loss = sum_factors / (self.factors[i] + eps) * layer_loss
            
            loss = loss + layer_loss
            
        return loss