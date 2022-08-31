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
def AT(x_array, temp):#atomic table representation
    #i='4 La$_{66}$Al$_{14}$Cu$_{10}$Ni$_{10}$ [c][15]'
    print(x_array)
    X= [[[0.0 for ai in range(18)]for aj in range(9)] for ak in range(2)]
    # gfa=re.findall('\[[a-c]?\]',i)[0]
    
    tx1_element=[element[i] for i in range(10) if x_array[i] != 0]
    tx2_value=[x_array[i] for i in range(10) if x_array[i] != 0]
    for j in range(len(tx2_value)):
        # index=int(property_list[element_name.index(tx1_element[j])][1])#atomic number Z
        index=new_index[int(property_list[element_name.index(tx1_element[j])][1])-1]
        xi=int(Z_row_column[index-1][1])#row num
        xj=int(Z_row_column[index-1][2])#col num
        X[0][xi-1][xj-1]=tx2_value[j]/100.0

    # X_BMG=copy.deepcopy(X)
    # X_BMG[0][10][10]=1.0 #processing parameter
    print(temp)
    X[1] = [[(temp - 273.15) / 2000 for ai in range(18)]for aj in range(9)]
    return X #[X,X_BMG]
    
Xs=[]
Ys=[]
for i in range(len(df)):
    # print(df.iloc[i,11])
    Xs.append(AT(df.iloc[i, 1:11], df.iloc[i, 11]))
    Ys.append(df.iloc[i, 12])

import numpy as np
Xs = np.array(Xs)
Ys = np.array(Ys)
print(Xs.shape)
print(Ys.shape)
pickle.dump([Xs, Ys], open('high_data_new.pkl', "wb"))