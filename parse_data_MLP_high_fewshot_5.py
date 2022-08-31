import pandas as pd
import pickle
import re
import copy
df = pd.read_csv("high_fidelity.csv", sep="\t")
[property_name_list,property_list,element_name,_] = pickle.load(open('element_property.txt', 'rb'))
element = list(df.columns[1:11])
print(element)

Z_row_column = pickle.load(open('Z_row_column.txt', 'rb'))
new_index=[int(i[4]) for i in Z_row_column]

    
Xs=[]
Ys=[]
import numpy as np
for i in range(len(df)):
    # print(df.iloc[i,11])
    print(df.iloc[i, 12])
    Xs.append(list(np.array(df.iloc[i, 1:12])))
    Ys.append(df.iloc[i, 12])

import numpy as np
Xs = np.array(Xs)
Ys = np.array(Ys)
print(Xs.shape)
print(Ys.shape)

import seaborn as sns
import matplotlib.pyplot as plt
# Create the default pairplot
sns.set_theme(style="white")
df = {}
for i in range(len(element)):
    df[element[i]] = Xs[:,i]

df['Temperature'] = Xs[:, -1]
df['YS'] = Ys
df = pd.DataFrame(df)
sns.pairplot(df)
plt.savefig("pairwise.png", bbox_inches="tight")
assert False

num = (np.sum((Xs != 0), 1)).astype(int) - 1
perm = np.random.permutation(240)

Xs = Xs[perm]
Ys = Ys[perm]
num = num[perm]
print(num)
count = {5: 5, 4: 4, 6: 1, 3: 1}

final_Xs = []
final_Ys = []

final_test_Xs = []
final_test_Ys = []

for i in range(len(Xs)):

    if count[num[i]] > 0:
        final_Xs.append(Xs[i])
        final_Ys.append(Ys[i])
        count[num[i]] -= 1
    else:
        final_test_Xs.append(Xs[i])
        final_test_Ys.append(Ys[i])

final_Xs = np.stack(final_Xs)
final_Ys = np.stack(final_Ys)

final_test_Xs = np.stack(final_test_Xs)
final_test_Ys = np.stack(final_test_Ys)


print(final_Xs)
print(final_Ys)

print(final_Xs.shape)
current_split = {"train_Xs": final_Xs, "train_labels": final_Ys, "test_Xs": final_test_Xs, "test_labels": final_test_Ys}
pickle.dump(current_split, open(f"data_split_MLP_5_percent.pkl", "wb"))
# pickle.dump([Xs, Ys], open('high_data.pkl', "wb"))

