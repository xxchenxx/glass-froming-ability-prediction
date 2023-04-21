import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--init", type=str, default='kn')

args = parser.parse_args()

def set_seed(seed):
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if args.init == 'zero':
                    nn.init.zeros_(m.weight)
                elif args.init == 'ku':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif args.init == 'kn':
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif args.init == 'xu':
                    nn.init.xavior_uniform_(m.weight)
            
    def forward(self, x):
        out = self.fc(x)
        return out
import pandas as pd
MSEs = []
for seed in range(10):
    set_seed(seed)
    model = Model()
    model = model.cuda()
    data = pickle.load(open(f"data_split_MLP.pkl", "rb"))
    y = data['train_labels']
    from sklearn.preprocessing import StandardScaler
    
    X_train = torch.from_numpy(data['train_Xs']).float()
    Y_train = torch.from_numpy(data['train_labels']).float()
    X_test = torch.from_numpy(data['test_Xs']).float()
    Y_test = torch.from_numpy(data['test_labels']).float()

    X = torch.cat([X_train, X_test], 0)
    Y = torch.cat([Y_train, Y_test], 0)

    X_train = X[:23]
    X_test = X[23:]

    Y_train = Y[:23]
    Y_test = Y[23:]
    # X_train = torch.cat([X_train, X_test], 0)
    # Y_train = torch.cat([Y_train, Y_test], 0)
    fit = StandardScaler().fit(X_train)
    # print(X_train.shape)
    X_train = torch.from_numpy(fit.transform(X_train.numpy())).float()
    X_test = torch.from_numpy(fit.transform(X_test.numpy())).float()
    # predict_data = np.zeros((12, 11))
    # predict_data[:6,:-1] = np.array([0, 0, 0, 21.7, 20.6, 15.6, 0, 21., 21.1, 0])
    # predict_data[6:,:-1] = np.array([0, 0, 0, 25.6, 22.7, 24.4, 0, 0, 27.3, 0])
    # predict_data[:6, -1] = np.array([873.15, 1073.15, 1273.15, 1473.15, 1673.15, 1873.15])
    # predict_data[6:, -1] = np.array([873.15, 1073.15, 1273.15, 1473.15, 1673.15, 1873.15])
    import pandas as pd
    predict_data = np.array(pd.read_csv('new_data.csv', header=1))
    Y_unseen = predict_data[:, -1]
    predict_data = predict_data[:, :-1] 
    predict_data[:, :-1] = predict_data[:, :-1] * 100
    X_unseen = torch.from_numpy(fit.transform(predict_data)).float()
    Y_unseen = torch.from_numpy(Y_unseen).float()
    # print(X_train)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    unseen_dataset = torch.utils.data.TensorDataset(X_unseen, Y_unseen)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)
    unseen_dataloader = torch.utils.data.DataLoader(unseen_dataset, batch_size=args.batch_size)

    optimizer = torch.optim.SGD([
                {'params': [p for name, p in model.named_parameters() if 'mask' not in name], "lr": args.lr, 'weight_decay': 0},
                {'params': [p for name, p in model.named_parameters() if 'mask' in name], "lr": 10, 'weight_decay': 0}
            ])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.1)

    best_mse = 10000000000
    best_corr = 0
    for epoch in range(90):
        model.train()
        # print(len(train_dataloader))
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
            # print(f"Epoch: [{epoch}], MSE: {mse}")

                unseen_Xs = []
                unseen_Ys = []
                for x, y in unseen_dataloader:
                    x = x.cuda()
                    out = model(x)
                    unseen_Xs.append(out.view(-1))
                    unseen_Ys.append(y.view(-1))
                unseen_Xs = torch.cat(unseen_Xs, 0).cpu()
                unseen_Ys = torch.cat(unseen_Ys, 0)
                # print(Xs)
    print(f"Seed: {seed}, Best MSE: {best_mse}")
    print(unseen_Xs)
    MSEs.append(best_mse)
    import matplotlib.pyplot as plt

    # plt.scatter(best_Xs.cpu().numpy(), best_Ys.cpu().numpy())
    # plt.savefig("regression_MLP.png", bbox_inches="tight")
    # plt.close()
    torch.save({
        'Xs': best_Xs,
        'Ys': best_Ys,
    }, f'MLP-{seed}.pkl')
print(np.mean(np.array(MSEs)))