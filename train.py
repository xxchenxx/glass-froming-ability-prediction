import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
import random
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
def set_seed(seed):
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(2, 2, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, 2, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, 2, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(576, 16),
            # nn.LayerNorm(16), 
            nn.ReLU(),
            nn.Linear(16, 1))
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.maxpool1(out)
        out = F.relu(self.conv2(out))
        out = self.maxpool2(out)
        out = F.relu(self.conv3(out))
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
    
for seed in range(1):
    set_seed(seed)
    model = Model()
    model = model.cuda()
    for j in range(10):
        data = pickle.load(open(f"data_split_{j}.pkl", "rb"))
        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data['train_Xs']).float(), torch.from_numpy(data['train_labels']).float())
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data['test_Xs']).float(), torch.from_numpy(data['test_labels']).float())

        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

        best_mse = 100000
        best_corr = 0
        for epoch in range(100):
            Xs = []
            Ys = []
            for x, y in train_dataloader:
                x = x.cuda()
                y = y.cuda()
                out = model(x)

                loss = F.mse_loss(out, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                Xs.append(out.detach().cpu().numpy())
                Ys.append(y.detach().cpu().numpy())
                
            Xs = np.concatenate(Xs, 0).reshape(-1)
            Ys = np.concatenate(Ys, 0)
            corr = pearsonr(Xs, Ys)[0]
            print(corr)
            lr_scheduler.step()
            with torch.no_grad():
                Xs = []
                Ys = []
                for x, y in test_dataloader:
                    x = x.cuda()
                    out = model(x)
                    Xs.append(out.cpu().numpy())
                    Ys.append(y.numpy())
                
                Xs = np.concatenate(Xs, 0).reshape(-1)
                Ys = np.concatenate(Ys, 0)
                mse = np.mean((Xs-Ys)**2)
                corr = pearsonr(Xs, Ys)[0]
                if best_mse > mse:
                    best_mse = mse
                    best_corr = corr
                # print(f"Epoch: [{epoch}], MSE: {np.mean((Xs-Ys)**2)}, Correlation: {pearsonr(Xs, Ys)[0]}")

        print(f"Split: {j}, Seed: {seed}, Best MSE: {mse}, correlation: {corr}")