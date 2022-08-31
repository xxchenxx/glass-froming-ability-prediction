import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
import random
from imblearn.over_sampling import RandomOverSampler
import smogn
from utils import pruning_model_linear
"""
def CNNmodel_PTR():
    np.random.seed(1)  # for reproducibility
    model = Sequential()
    # Conv layer 1 output shape (32, 28, 28)
    model.add(Convolution2D(batch_input_shape=(None, 1, 9, 18),filters=8,
        kernel_size=3,strides=1, padding='same',
        data_format='channels_first', activation='relu'))
    model.add(MaxPooling2D(pool_size=2,strides=2,padding='same',data_format='channels_first',))
    model.add(Convolution2D(16, 3, strides=1, padding='same', 
                            data_format='channels_first',activation='relu'))#16
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))
    model.add(Convolution2D(32, 3, strides=1, padding='same', 
                            data_format='channels_first', activation='relu'))
    model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))#32
    model.add(Flatten())
    model.add(Dense(2,activation='softmax'))
    adam = Adam(lr=1e-4)
    # We add metrics to get more results you want to see
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
    return model
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--init", type=str, default='zero')

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
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1))
            # nn.LayerNorm(16), 
            # nn.ReLU(),
            #nn.Linear(16, 1, bias=False))
        
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
set_seed(0)
model = Model()
model = model.cuda()

data = pickle.load(open(f"data_split_MLP_5_percent.pkl", "rb"))
y = data['train_labels']
from sklearn.preprocessing import StandardScaler

X_train = torch.from_numpy(data['train_Xs']).float()
Y_train = torch.from_numpy(data['train_labels']).float()
X_test = torch.from_numpy(data['test_Xs']).float()
Y_test = torch.from_numpy(data['test_labels']).float()
# X_train = torch.cat([X_train, X_test], 0)
# Y_train = torch.cat([Y_train, Y_test], 0)
fit = StandardScaler().fit(X_train)
# print(X_train.shape)
X_train = torch.from_numpy(fit.transform(X_train.numpy()))
X_test = torch.from_numpy(fit.transform(X_test.numpy()))
# print(X_train)
train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)

for round_ in range(10):
    
    
    optimizer = torch.optim.SGD([
                {'params': [p for name, p in model.named_parameters() if 'mask' not in name], "lr": args.lr, 'weight_decay': 0},
                {'params': [p for name, p in model.named_parameters() if 'mask' in name], "lr": 10, 'weight_decay': 0}
            ])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

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
                # print(epoch)
            # print(f"Epoch: [{epoch}], MSE: {mse}")
    print(f"Seed: {round_}, Best MSE: {best_mse}")
    # MSEs.append(best_mse)
    # import matplotlib.pyplot as plt
    # pruning_model_linear(model, 0.05)
    # plt.scatter(best_Xs.cpu().numpy(), best_Ys.cpu().numpy())
    # plt.savefig("regression_MLP.png", bbox_inches="tight")
    # plt.close()

print(np.mean(np.array(MSEs)))