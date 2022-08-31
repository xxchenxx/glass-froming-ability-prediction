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