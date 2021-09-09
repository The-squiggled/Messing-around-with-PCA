# -*- coding: utf-8 -*-
"""
Name: Principle component Analysis 400 emerging water contaminants

@author: annaf
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA



url_2 ="https://raw.githubusercontent.com/The-squiggled/Messing-around-with-PCA/main/result%20all%20labels.csv"
label_list = pd.read_csv(url_2)

url = "https://raw.githubusercontent.com/The-squiggled/Messing-around-with-PCA/main/result%20all.csv"

data_full = pd.read_csv(url)
data = data_full
features = list(data)
data['label'] = label_list
x = data.loc[:,features].values
x = StandardScaler().fit_transform(x)
print(np.std(x))

feat_cols = ['feature'+str(i) for i in range(x.shape[1])]
normalised_data = pd.DataFrame(x,columns=feat_cols)

pca_400 = PCA(n_components=2)
principalComponents_data = pca_400.fit_transform(x)

principal_data_Df = pd.DataFrame(data = principalComponents_data, 
                                 columns = ['principal component 1',
                                            'principal component 2'])
print('Explained variation per principal component: {}'.format(pca_400.explained_variance_ratio_))



plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Principal Component - 1',fontsize=20)
plt.ylabel('Principal Component - 2',fontsize=20)
plt.title("Principal Component Analysis",fontsize=20)
plt.scatter(principalComponents_data[:,0], principalComponents_data[:,1])
principal_data_Df['label'] = label_list
# =============================================================================
# targets = ['Benign', 'Malignant']
# colors = ['r', 'g']
# 
# for target, color in zip(targets,colors):
#     indicesToKeep = data['label'] == target
#     plt.scatter(principal_data_Df.loc[indicesToKeep, 'principal component 1']
#                , principal_data_Df.loc[indicesToKeep, 'principal component 2']
#                ,
#                c = color, s = 50)
# 
# plt.legend(targets,prop={'size': 15})
# =============================================================================
