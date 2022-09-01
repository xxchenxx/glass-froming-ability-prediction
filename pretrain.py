import pickle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument('--lf-data', default='low_data_split.pkl', type=str, help='low-fidelity data path')
parser.add_argument('--data-dir', type=str, default='data')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--learning-rate', type=float, default=1e-4)

args = parser.parse_args()

def main(args):
    seed = args.seed
    set_seed(seed)
    model = Model()
    model = model.cuda()
    data = pickle.load(open(os.path.join(args.data_dir, args.lf_data), "rb"))
    
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data['train_Xs']).float(), torch.from_numpy(data['train_labels']).float())
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data['test_Xs']).float(), torch.from_numpy(data['test_labels']).float())

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)

    best_mse = 100000
    best_corr = 0
    for epoch in range(args.epoch):
        model.train()
        for x, y in train_dataloader:
            x = x.cuda()
            y = y.cuda()
            out = model(x)

            loss = F.mse_loss(out.squeeze(), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        lr_scheduler.step()
        model.eval()
        with torch.no_grad():
            Xs = []
            Ys = []
            for x, y in test_dataloader:
                x = x.cuda()
                out = model(x)
                Xs.append(out.view(-1))
                Ys.append(y.view(-1))
            
            Xs = torch.cat(Xs, 0).cpu()
            Ys = torch.cat(Ys, 0).cpu()
            mse = F.mse_loss(Xs, Ys)
            if best_mse > mse:
                best_mse = mse
                best_state_dict = model.state_dict()
                torch.save(best_state_dict, "low_pretrained_regression.pth.tar")
            print(f"Epoch: [{epoch}], Testing MSE: {mse}")

if __name__ == "__main__":
    main(args)