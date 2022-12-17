import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import random
import numpy as np
import torch.nn.functional as F


def pruning_model(model, px, conv1=True):

    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if name == 'conv1':
                if conv1:
                    parameters_to_prune.append((m,'weight'))
                else:
                    print('skip conv1 for L1 unstructure global pruning')
            else:
                parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def pruning_model_linear(model, px, conv1=True):

    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Linear):
            parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def set_seed(seed):
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

import torch
import torch.nn as nn


class MaskedConv2d(nn.Conv2d):
    def set_incremental_weights(self, beta=True) -> None:
        # self.register_parameter('weight_beta', torch.nn.Parameter(torch.zeros_like(self.weight.data)))
        self.register_parameter('mask_alpha', torch.nn.Parameter(torch.ones_like(self.weight.data)))
        self.weight.requires_grad = True
        self.mode = 'lower'
        self.epsilon = 0.1
        self.beta = beta
    
    def set_lower(self) -> None:
        self.weight.requires_grad = True
        # self.weight_beta.requires_grad = True
        self.mask_alpha.requires_grad = True
        self.mode = 'lower'
    
    def set_upper(self) -> None:
        self.weight.requires_grad = True
        # self.weight_beta.requires_grad = True
        self.mask_alpha.requires_grad = True
        self.mode = 'upper'

    def forward(self, input):
        weight = (self.weight) * (self.mask_alpha ** 2) / (self.mask_alpha ** 2 + self.epsilon)

        #print(weight)
        if torch.__version__ > "1.7.1":
            return self._conv_forward(input, weight, self.bias) 
        else:
            return self._conv_forward(input, weight) 


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = MaskedConv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool1 = nn.MaxPool2d(2, 2, padding=1)
        self.conv2 = MaskedConv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(2, 2, padding=1)
        self.conv3 = MaskedConv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.maxpool3 = nn.MaxPool2d(2, 2, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(1152, 64),
            # nn.LayerNorm(64), 
            nn.ReLU(),
            nn.Linear(64, 1, bias=False))
        
        for m in self.modules():
            if isinstance(m, MaskedConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
    def forward(self, x, second=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.maxpool2(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = out.view(out.shape[0], -1)
        if not second:
            out = self.fc(out)
            return out
        else:
            return self.fc(out), self.new_fc(out)