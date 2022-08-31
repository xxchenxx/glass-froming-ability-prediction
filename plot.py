import pickle
a = pickle.load(open("data_split_5.pkl", "rb"))
print(a['test_labels'])
print(a['train_labels'])
