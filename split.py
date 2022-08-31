import torch
import pandas 
import pickle
import numpy as np
X, Y = pickle.load(open("high_data_new.pkl", "rb"))
X.shape
Y.shape
from sklearn.model_selection import KFold
import torch

import pickle
kfold = KFold(n_splits=10, shuffle=True)


for split_id, (train_id, test_id) in enumerate(kfold.split(X, Y)):
    current_split = {"train_Xs": X[train_id], "train_labels": Y[train_id], "test_Xs": X[test_id], "test_labels": Y[test_id]}
    pickle.dump(current_split, open(f"data_split_new_{split_id}.pkl", "wb"))
