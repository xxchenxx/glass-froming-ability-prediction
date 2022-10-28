import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
import random
import wandb 
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
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['mse'])
    return model
"""
def set_seed(seed):
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class MoEConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, n_experts=3):
        super().__init__()
        print(in_channels, out_channels, kernel_size, stride, padding)
        self.n_experts = n_experts
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding) for _ in range(n_experts)
        ])
    
    def forward(self, x, idx):
        results = torch.stack([self.convs[i](x) for i in range(self.n_experts)], 1)

        idx = idx.view(-1, 1, 1, 1, 1).repeat((1, 1, *results.shape[2:]))
        results = torch.squeeze(torch.gather(results, 1, idx), 1)

        return results

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = MoEConv(2, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool1 = nn.MaxPool2d(2, 2, padding=1)
        self.conv2 = MoEConv(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(2, 2, padding=1)
        self.conv3 = MoEConv(32, 64, 3, padding=1)
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
            
    def forward(self, x, idx):
        out = F.relu(self.bn1(self.conv1(x, idx)))
        out = self.maxpool1(out)
        out = F.relu(self.bn2(self.conv2(out, idx)))
        out = self.maxpool2(out)
        out = F.relu(self.bn3(self.conv3(out, idx)))
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out
    
for seed in range(1):
    set_seed(seed)
    model = Model()
    model = model.cuda()
    data = pickle.load(open(f"data/low_data_split.pkl", "rb"))
    
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data['train_Xs']).float(), torch.from_numpy(data['train_labels']).float())
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data['test_Xs']).float(), torch.from_numpy(data['test_labels']).float())

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)


    best_mse = 100000
    best_corr = 0
    steps = 0
    wandb.init(project='llnl', entity='xxchen', name='no_cl')
    for curriculum in range(5,6):

        for epoch in range(180):
            model.train()
            for x, y in train_dataloader:
                x = x.cuda()
                temp = (x[:, 1, 0, 0] * 3).long()

                x = x
                if x.shape[0] == 0:
                    continue
                y = y.cuda()
                out = model(x, temp)

                loss = F.mse_loss(out.view(-1), y.view(-1))
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
                    y = y.view(-1)
                    out = model(x)
                    Xs.append(out.view(-1))
                    Ys.append(y.view(-1))
                
                Xs = torch.cat(Xs, 0).cpu()
                Ys = torch.cat(Ys, 0).cpu()
                mse = F.mse_loss(Xs.view(-1), Ys.view(-1))
                pearson = pearsonr(Xs, Ys)[0]
                wandb.log({"pearson": pearson, "test_loss": mse})
                if best_mse > mse:
                    best_mse = mse
                    best_state_dict = model.state_dict()
                    torch.save(best_state_dict, "low_pretrained_regression_10.pkl")
                print(f"Epoch: [{epoch}], MSE: {mse}")
        print(f"Seed: {seed}, Best MSE: {best_mse}")