import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
from imblearn.over_sampling import RandomOverSampler

from utils import MaskedConv2d,  pruning_model, set_seed



import pandas as pd

lower_steps = 1
no_alpha = no_beta = False

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


for seed in range(10):
    set_seed(seed)
    model = Model()
    model = model.cuda()
    
    data = pickle.load(open(f"data_split_10_percent.pkl", "rb"))
    y = data['train_labels']
    new_y = (y*2).astype(int)
    X = data['train_Xs']

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data['train_Xs']).float(), torch.from_numpy(data['train_labels']).float())
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data['test_Xs']).float(), torch.from_numpy(data['test_labels']).float())

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4)

    low_data = pickle.load(open(f"low_data_split.pkl", "rb"))
    low_train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(low_data['train_Xs']).float(), torch.from_numpy(low_data['train_labels']).float())
    low_test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(low_data['test_Xs']).float(), torch.from_numpy(low_data['test_labels']).float())

    low_train_dataloader = torch.utils.data.DataLoader(low_train_dataset, batch_size=4, shuffle=True)
    low_test_dataloader = torch.utils.data.DataLoader(low_test_dataset, batch_size=4)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_mse = 100000

    model.load_state_dict(torch.load("low_pretrained_regression.pkl"))
    import copy
    model.new_fc = nn.Sequential(
            nn.Linear(1152, 64),
            nn.ReLU(),
            nn.Linear(64, 1, bias=False)).cuda()

    model_lower = copy.deepcopy(model)
    # init pretrianed weight
    for m in model.modules():
        if isinstance(m, MaskedConv2d):
            m.set_incremental_weights()
    for m in model_lower.modules():
        if isinstance(m, MaskedConv2d):
            m.set_incremental_weights(beta=False)

    
    optimizer = torch.optim.SGD([
                {'params': [p for name, p in model.named_parameters() if 'mask' not in name], "lr": 0.001, 'weight_decay': 0},
                {'params': [p for name, p in model.named_parameters() if 'mask' in name], "lr": 10, 'weight_decay': 0}
            ])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    for epoch in range(90):
        Xs = []
        Ys = []
        model.train()
        for name, m in model.named_modules():
            if isinstance(m, MaskedConv2d):
                m.epsilon *= 0.9
        
        for x, y in train_dataloader:
        
            x = x.cuda()
            y = y.cuda()

            state_dict = model.state_dict()
            for key in list(state_dict.keys()):
                if 'mask_beta' in key: del state_dict[key]
            model_lower.load_state_dict(state_dict)
            weights = []
            alphas = []
            model.train()
            # print(y)
            out = model(x)
            loss = F.mse_loss(out, y)
            loss.backward()
            for name, m in model.named_modules():
                if isinstance(m, MaskedConv2d):
                    m.mask_alpha.grad = None
            # end unrolling
            optimizer.step()
            # calculate (a + b)
            model.zero_grad()

            loss = loss.float()
            for name, m in model.named_modules():
                if isinstance(m, MaskedConv2d):
                    if not no_beta:
                        beta = m.mask_beta.data.detach().clone()
                        lr = optimizer.param_groups[1]['lr']
                        # print(lr * 1e-4)
                        #print(beta.data.abs().mean())
                        
                        m1 = beta >= lr * 1e-4
                        m2 = beta <= -lr * 1e-4
                        m3 = (beta.abs() < lr * 1e-4)
                        m.mask_beta.data[m1] = m.mask_beta.data[m1] - lr * 1e-4
                        m.mask_beta.data[m2] = m.mask_beta.data[m2] + lr * 1e-4
                        m.mask_beta.data[m3] = 0
            Xs.append(out.detach().cpu().numpy())
            Ys.append(y.cpu().numpy())
        # Xs = np.concatenate(Xs, 0).reshape(-1)
        # Ys = np.concatenate(Ys, 0)
        # mse = np.mean((Xs-Ys)**2)
        # print(mse)
        # print(loss)
        lr_scheduler.step()
        model.eval()
        if epoch == 89:
            for name, m in model.named_modules():
                if isinstance(m, MaskedConv2d):
                    m.set_upper()
                    # print(((m.mask_beta ** 2) / ((m.mask_beta ** 2) + m.epsilon)).mean())
                    # print(((m.mask_alpha ** 2) / ((m.mask_alpha ** 2) + m.epsilon)).mean())
                    print(((m.mask_alpha ** 2) / ((m.mask_alpha ** 2) + m.epsilon) * (m.mask_beta ** 2) / ((m.mask_beta ** 2) + m.epsilon)).mean())
                #print(m.mask_beta.data.abs().mean())
                #print(m.mask_alpha.data.abs().mean())

        with torch.no_grad():
            Xs = []
            Ys = []
            for x, y in test_dataloader:
                x = x.cuda()
                out = model(x)
                Xs.append(out.cpu().view(-1))
                Ys.append(y.view(-1,1))
            
            Xs = torch.cat(Xs, 0).view(-1)
            Ys = torch.cat(Ys, 0).view(-1)
            mse = F.mse_loss(Xs, Ys)
            if mse < best_mse:
                best_mse = mse
                best_Xs = Xs
                best_Ys = Ys

    print(best_mse)
    # import matplotlib.pyplot as plt
    # plt.scatter(best_Xs.cpu().numpy(), best_Ys.cpu().numpy())
    # plt.savefig("best_regression.png", bbox_inches="tight")
    # plt.close()
    # assert False