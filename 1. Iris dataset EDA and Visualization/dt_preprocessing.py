from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd

data = pd.read_csv('iris.csv')
print(data)
print(data.head())
print(data.shape)
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
print(x.shape)
clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(x, y)
print(clf.feature_importances_) 
model = SelectFromModel(clf, prefit=True)
x_new = model.transform(x)
print(x_new.shape) 
print(x_new)
x_new = pd.DataFrame(data=x_new,columns=["One","Two"])
y = pd.DataFrame(data=y,columns=["Species"])
final = pd.concat([x_new,y],axis=1)
print(final)

