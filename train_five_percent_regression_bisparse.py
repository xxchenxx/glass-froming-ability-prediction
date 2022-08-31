import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import MaskedConv2d, set_seed
from models import Model, MaskedConv2d


import pandas as pd

lower_steps = 1
no_alpha = no_beta = False

extra_balance = 1

for seed in range(10):
    set_seed(seed)
    model = Model()
    model = model.cuda()
    
    data = pickle.load(open(f"data_split_10_percent.pkl", "rb"))
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data['train_Xs']).float(), torch.from_numpy(data['train_labels']).float())
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(data['test_Xs']).float(), torch.from_numpy(data['test_labels']).float())

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4)

    low_data = pickle.load(open(f"low_data_split.pkl", "rb"))
    low_train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(low_data['train_Xs']).float(), torch.from_numpy(low_data['train_labels']).float())
    low_test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(low_data['test_Xs']).float(), torch.from_numpy(low_data['test_labels']).float())

    low_train_dataloader = torch.utils.data.DataLoader(low_train_dataset, batch_size=4, shuffle=True)
    low_test_dataloader = torch.utils.data.DataLoader(low_test_dataset, batch_size=4)

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
            for name, m in model.named_modules():
                if isinstance(m, MaskedConv2d):
                    m.set_lower()

            for _ in range(lower_steps):   
                try:
                    low_image, low_target = next(low_train_dataloader_iter)
                except:
                    low_train_dataloader_iter = iter(low_train_dataloader)
                    low_image, low_target = next(low_train_dataloader_iter)
                # compute output
                low_image = low_image.cuda()
                low_target = low_target.cuda()
                
                
                output_old, output_new = model(low_image, second=True)
                loss = F.mse_loss(output_old.view(-1), low_target.view(-1))
                loss.backward()
                
                for name, m in model.named_modules():
                    if isinstance(m, MaskedConv2d):
                        m.mask_alpha.grad = None
                        m.mask_beta.grad = None

                if _ > 0:
                    for name, m in model.named_modules():
                        if isinstance(m, MaskedConv2d):
                            m.weight.grad.data = torch.sign(m.weight.grad.data)
                optimizer.step()
                optimizer.zero_grad()

                if _ == 0:
                    output_old, _ = model_lower(low_image, second=True)
                    loss_lower = F.mse_loss(output_old, low_target)
                    for name, m in model_lower.named_modules():
                        if isinstance(m, MaskedConv2d):
                            weights.append(m.weight)
                            alphas.append(m.mask_alpha)
                    grad_w = torch.autograd.grad(loss_lower, weights, create_graph=True, retain_graph=True)
                
            for name, m in model.named_modules():
                if isinstance(m, MaskedConv2d):
                    m.set_upper()

            out = model(x)
            loss = F.mse_loss(out, y)
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
                        m.mask_alpha.grad.data.sub_(grads[idx] * alpha_lr * extra_balance)
                        idx += 1

            optimizer.step()
            model.zero_grad()

            loss = loss.float()
            for name, m in model.named_modules():
                if isinstance(m, MaskedConv2d):
                    if not no_beta:
                        beta = m.mask_beta.data.detach().clone()
                        lr = optimizer.param_groups[1]['lr']                    
                        m1 = beta >= lr * 1e-4
                        m2 = beta <= -lr * 1e-4
                        m3 = (beta.abs() < lr * 1e-4)
                        m.mask_beta.data[m1] = m.mask_beta.data[m1] - lr * 1e-4
                        m.mask_beta.data[m2] = m.mask_beta.data[m2] + lr * 1e-4
                        m.mask_beta.data[m3] = 0
                    if not no_alpha:
                        alpha = m.mask_alpha.data.detach().clone()
                        lr = optimizer.param_groups[1]['lr']
                        # print(lr * 1e-4)
                        #print(alpha.data.abs().mean())
                        m1 = alpha >= lr * 1e-4
                        m2 = alpha <= -lr * 1e-4
                        m3 = (alpha.abs() < lr * 1e-4)
                        m.mask_alpha.data[m1] = m.mask_alpha.data[m1] - lr * 1e-4
                        m.mask_alpha.data[m2] = m.mask_alpha.data[m2] + lr * 1e-4
                        m.mask_alpha.data[m3] = 0
            Xs.append(out.detach().cpu().numpy())
            Ys.append(y.cpu().numpy())

        lr_scheduler.step()
        model.eval()
        if epoch == 89:
            for name, m in model.named_modules():
                if isinstance(m, MaskedConv2d):
                    m.set_upper()
                    print(((m.mask_alpha ** 2) / ((m.mask_alpha ** 2) + m.epsilon) * (m.mask_beta ** 2) / ((m.mask_beta ** 2) + m.epsilon)).mean())

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
    import matplotlib.pyplot as plt
    plt.scatter(best_Xs.cpu().numpy(), best_Ys.cpu().numpy())
    plt.savefig("best_regression.png", bbox_inches="tight")
    plt.close()
    # assert False