import pickle
from re import X

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
import random

def set_seed(seed):
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool1 = nn.MaxPool2d(2, 2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(2, 2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.maxpool3 = nn.MaxPool2d(2, 2, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(1152, 64),
            # nn.LayerNorm(64), 
            nn.ReLU(),
            nn.Linear(64, 1, bias=False))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.maxpool2(out)
        out = F.relu(self.bn3(self.conv3(out)))
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
    

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--output-name", type=str)

args = parser.parse_args()

for seed in range(42,43):
    set_seed(seed)
    model = Model()
    model = model.cuda()
    data = pickle.load(open(f"{args.data}", "rb"))

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    y = data['train_labels']
    
    X_train = torch.from_numpy(data['train_Xs']).float()
    Y_train = torch.from_numpy(data['train_labels']).float()
    X_test = torch.from_numpy(data['test_Xs']).float()
    Y_test = torch.from_numpy(data['test_labels']).float()

    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    optimizer = torch.optim.SGD([
                {'params': [p for name, p in model.named_parameters() if 'mask' not in name], "lr": args.lr, 'weight_decay': 0},
                {'params': [p for name, p in model.named_parameters() if 'mask' in name], "lr": 10, 'weight_decay': 0}
            ])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_mse = 100000
    best_corr = 0
    for epoch in range(90):
        model.train()
        # print(len(train_dataloader))
        for x, y in train_dataloader:
            x = x.cuda()
            y = y.cuda().view(-1)
            out = model(x).view(-1)

            loss = F.mse_loss(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        lr_scheduler.step()
        model.eval()
        with torch.no_grad():
            Xs = []
            Ys = []
            # print(len(test_dataloader))
            for x, y in test_dataloader:
                x = x.cuda()
                out = model(x)
                Xs.append(out.view(-1))
                Ys.append(y.view(-1))
            
            Xs = torch.cat(Xs, 0).cpu()
            Ys = torch.cat(Ys, 0)
            mse = F.mse_loss(Xs, Ys)
            if best_mse > mse:
                best_mse = mse
                best_Xs = Xs
                best_Ys = Ys
                best_model_weight = model.state_dict()
            # print(f"Epoch: [{epoch}], MSE: {mse}")
    print(f"Seed: {seed}, Best MSE: {best_mse}")
    
    import matplotlib.pyplot as plt

    plt.scatter(best_Xs.cpu().numpy(), best_Ys.cpu().numpy())
    plt.savefig("regression.png", bbox_inches="tight")
    plt.close()
    torch.save({
        'Xs': best_Xs,
        'Ys': best_Ys,
        'model_weight': best_model_weight
    }, f'{args.output_name}-{seed}.pkl')

