import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr
from imblearn.over_sampling import RandomOverSampler

from utils import MaskedConv2d, Model, pruning_model, set_seed


import sys
alr = float(sys.argv[1])
print(alr)
import pandas as pd

lower_steps = 1
no_alpha = no_beta = False

for seed in range(10):
    model = Model()
    model = model.cuda()
    
    data = pickle.load(open(f"data_split_new_{seed}.pkl", "rb"))
    y = data['train_labels']
    new_y = (y*2).astype(int)
    X = data['train_Xs']
    upsample = False
    print(f"Upsample: {upsample}")
    if upsample:
        ros = RandomOverSampler(random_state=0)
        X = X.reshape(X.shape[0], -1)
        # print(X.shape)
        # print(y.reshape(-1,1).shape)
        X = np.concatenate([X, y.reshape(-1,1)], 1)
        X_resampled, y_resampled = ros.fit_resample(X, new_y)
        # print(X_resampled.shape)
        y_original = X_resampled[:, -1]
        # print(y_resampled)
        X_original = X_resampled[:, :-1].reshape(-1, *data['train_Xs'].shape[1:])
    else:
        X_original = X
        y_resampled = new_y

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_original).float(), torch.from_numpy(y_resampled).long())
    fake_test_label = (data['test_labels'] * 2).astype(int)
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data['test_Xs']).float(), torch.from_numpy(fake_test_label).long())

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4)

    low_data = pickle.load(open(f"low_data_split.pkl", "rb"))
    low_y = low_data['train_labels']
    low_y = (low_y*2).astype(int)
    low_y[low_y > 4] = 4
    low_test_y = (low_data['test_labels'] * 2).astype(int)
    low_test_y[low_test_y > 4] = 4
    low_train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(low_data['train_Xs']).float(), torch.from_numpy(low_y).long())
    low_test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(low_data['test_Xs']).float(), torch.from_numpy(low_test_y).long())

    low_train_dataloader = torch.utils.data.DataLoader(low_train_dataset, batch_size=4, shuffle=True)
    low_test_dataloader = torch.utils.data.DataLoader(low_test_dataset, batch_size=4)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_acc = 0

    model.load_state_dict(torch.load("low_pretrained.pkl"))
    import copy
    model.new_fc = nn.Sequential(
            nn.Linear(1152, 64),
            # nn.LayerNorm(64), 
            nn.ReLU(),
            nn.Linear(64, 5, bias=False)).cuda()

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
                {'params': [p for name, p in model.named_parameters() if 'mask' in name], "lr": alr, 'weight_decay': 0}
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

            model.eval()
            for _ in range(1):     
                previous_lr = optimizer.param_groups[0]['lr']
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']
                try:
                    low_image, low_target = next(low_train_dataloader_iter)
                except:
                    low_train_dataloader_iter = iter(low_train_dataloader)
                    low_image, low_target = next(low_train_dataloader_iter)
                # compute output
                low_image = low_image.cuda()
                low_target = low_target.cuda()
                
                
                output_old, output_new = model(low_image, second=True)
                loss = F.cross_entropy(output_old, low_target)
                # print(loss)
                loss.backward()
                
                for name, m in model.named_modules():
                    if isinstance(m, MaskedConv2d):
                        m.mask_alpha.grad = None
                if _ > 0:
                    for name, m in model.named_modules():
                        if isinstance(m, MaskedConv2d):
                            m.weight.grad.data = torch.sign(m.weight.grad.data)
                optimizer.step()
                optimizer.zero_grad()
                

                if _ == 0:
                    output_old, _ = model_lower(low_image, second=True)
                    loss_lower = F.cross_entropy(output_old, low_target)
                    for name, m in model_lower.named_modules():
                        if isinstance(m, MaskedConv2d):
                            weights.append(m.weight)
                            alphas.append(m.mask_alpha)
                    grad_w = torch.autograd.grad(loss_lower, weights, create_graph=True, retain_graph=True)

                optimizer.param_groups[0]['lr'] = previous_lr

            for name, m in model.named_modules():
                if isinstance(m, MaskedConv2d):
                    m.set_upper()
            model.train()
            # print(y)
            out, _ = model(x, second=True)
            loss = F.cross_entropy(out, y)
            loss.backward()

            grad_new_w = []
            for name, m in model.named_modules():
                if isinstance(m, MaskedConv2d):
                    grad_new_w.append(m.weight.grad.data.clone())
            

            aux_loss = 0
            for go, gn in zip(grad_w, grad_new_w):
                aux_loss = aux_loss + torch.sum(go * gn)
            
            grads = torch.autograd.grad(aux_loss, alphas, retain_graph=True)
            idx = 0
            alpha_lr = optimizer.param_groups[0]['lr']
            if not no_alpha:
                for m in model.modules():
                    if isinstance(m, MaskedConv2d):
                        m.mask_alpha.data.sub_(grads[idx] * alpha_lr)
                        idx += 1
            optimizer.step()
            model.zero_grad()
            # end unrolling

            loss = loss.float()
            for name, m in model.named_modules():
                if isinstance(m, MaskedConv2d):
                    if not no_beta:
                        beta = m.mask_beta.data.detach().clone()
                        lr = optimizer.param_groups[1]['lr']
                        # print(lr * 1e-4)
                        #print(beta.data.abs().mean())
                        
                        m1 = beta >= lr * 1.42e-3
                        m2 = beta <= -lr * 1.42e-3
                        m3 = (beta.abs() < lr * 1.42e-3)
                        m.mask_beta.data[m1] = m.mask_beta.data[m1] - lr * 1.42e-3
                        m.mask_beta.data[m2] = m.mask_beta.data[m2] + lr * 1.42e-3
                        m.mask_beta.data[m3] = 0
                    if not no_alpha:
                        alpha = m.mask_alpha.data.detach().clone()
                        lr = optimizer.param_groups[1]['lr']
                        # print(lr * 1.42e-3)
                        #print(alpha.data.abs().mean())
                        m1 = alpha >= lr * 1.42e-3
                        m2 = alpha <= -lr * 1.42e-3
                        m3 = (alpha.abs() < lr * 1.42e-3)
                        m.mask_alpha.data[m1] = m.mask_alpha.data[m1] - lr * 1.42e-3
                        m.mask_alpha.data[m2] = m.mask_alpha.data[m2] + lr * 1.42e-3
                        m.mask_alpha.data[m3] = 0

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
                out, _ = model(x, second=True)
                Xs.append(torch.argmax(out,1).cpu().view(-1))
                Ys.append(y.view(-1,1))
            
            Xs = torch.cat(Xs, 0).view(-1)
            Ys = torch.cat(Ys, 0).view(-1)
            acc = ((Xs==Ys).sum()/Xs.shape[0])
            # print(acc)
            if acc > best_acc:
                best_acc = acc
    print(best_acc)

