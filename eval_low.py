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
    
for seed in range(1):
    set_seed(seed)
    model = Model()
    model = model.cuda()
    data = pickle.load(open(f"low_data_split.pkl", "rb"))
    y = data['train_labels']
    new_y = (y*2).astype(int)
    new_y[new_y > 4] = 4
    new_test_y = (data['test_labels'] * 2).astype(int)
    new_test_y[new_test_y > 4] = 4
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data['train_Xs']).float(), torch.from_numpy(new_y).long())
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data['test_Xs']).float(), torch.from_numpy(new_test_y).long())

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    best_mse = 100000
    best_corr = 0
    best_accuracy = 0
    model.load_state_dict(torch.load('low_pretrained.pkl'))
    for epoch in range(1):
        model.eval()
        with torch.no_grad():
            Xs = []
            Ys = []
            for x, y in test_dataloader:
                x = x.cuda()
                out = model(x)
                Xs.append(torch.argmax(out, 1).cpu())
                Ys.append(y.view(-1))
            
            Xs = torch.cat(Xs, 0)
            Ys = torch.cat(Ys, 0)
            accuracy = (Xs == Ys).sum() / Xs.shape[0]
            print(f"Epoch: [{epoch}], Acc: {accuracy}")
        
        with torch.no_grad():
            Xs = []
            Ys = []
            for x, y in train_dataloader:
                x = x.cuda()
                out = model(x)
                Xs.append(torch.argmax(out, 1).cpu())
                Ys.append(y.view(-1))
            
            Xs = torch.cat(Xs, 0)
            Ys = torch.cat(Ys, 0)
            accuracy = (Xs == Ys).sum() / Xs.shape[0]
            print(f"Epoch: [{epoch}], Acc: {accuracy}")
    print(f"Seed: {seed}, Best Acc: {best_accuracy}")