import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
import random
from imblearn.over_sampling import RandomOverSampler
import smogn
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
            nn.Linear(64, 5, bias=False))
        
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
import pandas as pd
for seed in range(10):
    set_seed(seed)
    model = Model()
    model = model.cuda()
    data = pickle.load(open(f"data_split_10_percent.pkl", "rb"))
    # y = np.concatenate([data['train_labels'], data['test_labels']], 0)
    # X = np.concatenate([data['train_Xs'], data['test_Xs']], 0)
    y = data['train_labels']
    new_y = (y*2).astype(int)
    X = data['train_Xs']
    upsample = True
    if upsample:
        ros = RandomOverSampler(random_state=0)
        X = X.reshape(X.shape[0], -1)
        # print(X.shape)
        # print(y.reshape(-1,1).shape)
        X = np.concatenate([X, y.reshape(-1,1)], 1)
        X_resampled, y_resampled = ros.fit_resample(X, new_y)
        print(X_resampled.shape)
        y_original = X_resampled[:, -1]
        print(y_resampled)
        X_original = X_resampled[:, :-1].reshape(-1, *data['train_Xs'].shape[1:])
    else:
        X_original = X
        y_resampled = new_y
    
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_original).float(), torch.from_numpy(y_resampled).long())
    fake_test_label = (data['test_labels'] * 2).astype(int)
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data['test_Xs']).float(), torch.from_numpy(fake_test_label).long())

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4)

    optimizer = torch.optim.SGD([
                {'params': [p for name, p in model.named_parameters() if 'mask' not in name], "lr": 0.01, 'weight_decay': 0},
                {'params': [p for name, p in model.named_parameters() if 'mask' in name], "lr": 10, 'weight_decay': 0}
            ])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    best_acc = 0
    print(len(fake_test_label))
    for epoch in range(70):
        Xs = []
        Ys = []
        model.train()
        for x, y in train_dataloader:
            x = x.cuda()
            y = y.cuda()
            # print(y)
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            Xs.append(out.detach().cpu().numpy())
            Ys.append(y.cpu().numpy())
        Xs = np.concatenate(Xs, 0).reshape(-1)
        Ys = np.concatenate(Ys, 0)
        # mse = np.mean((Xs-Ys)**2)
        # print(mse)
        # print(loss)
        lr_scheduler.step()
        model.eval()
        with torch.no_grad():
            Xs = []
            Ys = []
            for x, y in test_dataloader:
                x = x.cuda()
                out = model(x)
                Xs.append(torch.argmax(out,1).cpu().view(-1))
                Ys.append(y.view(-1,1))
            
            Xs = torch.cat(Xs, 0).view(-1)
            Ys = torch.cat(Ys, 0).view(-1)
            acc = ((Xs==Ys).sum()/Xs.shape[0])
            # print(acc)
            if acc > best_acc:
                best_acc = acc
            # for i in range(5):
            #     print(f"{i}:", (Ys == i).sum())
            #     print(f"{i}:", ((Xs==Ys) * (Ys == i)).sum() / (Ys == i).sum())
            # mse = np.mean((Xs-Ys)**2)
            # corr = pearsonr(Xs, Ys)[0]
            #if best_mse > mse:
            #    best_mse = mse
            #     best_corr = corr
            #    best_Xs = Xs
            #    best_Ys = Ys
            # print(f"Epoch: [{epoch}], MSE: {np.mean((Xs-Ys)**2)}, Correlation: {pearsonr(Xs, Ys)[0]}")

    # print(f"Seed: {seed}, Best MSE: {mse}, correlation: {corr}")
    print(best_acc)
# import matplotlib.pyplot as plt
# plt.scatter(best_Xs, best_Ys)
# plt.plot(range(len(best_Xs)), , label="Ground-truth")
# plt.legend()
# plt.savefig("prediction.png", bbox_inches="tight")