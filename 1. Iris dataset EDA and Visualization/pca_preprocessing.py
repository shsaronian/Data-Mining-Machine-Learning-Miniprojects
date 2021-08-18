# -*- coding: utf-8 -*-
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('iris.csv')
print(data)
print(data.head())
print(data.shape)
print(data.ndim)
print(data.describe())
print(data.mode( numeric_only=True))
print(data.info())
print(data.isnull().sum())
print(data["Species"].unique())
print(data["Species"].value_counts())
print(data.groupby('Species').count())

data = pd.DataFrame(data=data.values,columns=["Sepal Length","Sepal Width","Petal Length","Petal Width","Species"])
features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
x = data.loc[:, features].values
y = data.loc[:,['Species']].values
print(x)
print(y)
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents  , columns = ['principal component 1', 'principal component 2'])
print(principalDf)
finalDf = pd.concat([principalDf, data[['Species']]], axis = 1)
print(finalDf)
print(pca.explained_variance_ratio_.sum())

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Setosa', 'Versicolor', 'Virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Species'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
