# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pylab 
import scipy.stats as stats

data = pd.read_csv('iris.csv')
print(data)
print(data.head())
print(data.shape)
print(data.describe())
print(data.info())
print(data.isnull().sum())
print(data["Species"].unique())
print(data["Species"].value_counts())
print(data.groupby('Species').count())

cc= data.corr(method='pearson')
print(cc)

cov= data.cov()
print(cov)

pd.plotting.parallel_coordinates(data, "Species", color=('#556270', '#4ECDC4', '#C7F464'))
plt.show()

measurements = np.random.normal(loc = 20, scale = 5, size=100)   
stats.probplot(measurements, dist="norm", plot=pylab)
pylab.show()

plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
sns.boxplot(x="Species",y="Sepal Length",data=data)
plt.subplot(2,2,2)
sns.boxplot(x="Species",y="Sepal Width",data=data)
plt.subplot(2,2,3)
sns.boxplot(x="Species",y="Petal Length",data=data)
plt.subplot(2,2,4)
sns.boxplot(x="Species",y="Petal Width",data=data)

data.hist(figsize=(10,11))
plt.figure()

sns.pairplot(data)
plt.figure(figsize=(10,11))
sns.heatmap(data.corr(),annot=True)
plt.plot()

sns.pairplot(data,hue="Species",diag_kind="kde")

plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
sns.violinplot(x="Species",y="Sepal Length",data=data)
plt.subplot(2,2,2)
sns.violinplot(x="Species",y="Sepal Width",data=data)
plt.subplot(2,2,3)
sns.violinplot(x="Species",y="Petal Length",data=data)
plt.subplot(2,2,4)
sns.violinplot(x="Species",y="Petal Width",data=data)

