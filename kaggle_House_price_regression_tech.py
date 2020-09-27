# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:20:45 2019

@author: Shuhei
"""
#どのfeatureを使うか様々な指標や図示によってデータの特徴を掴んでいく、という流れになります。
#欠損値の確認
print(train_df.isnull().sum())
#correlation
train_df.corr()

plt.plot(train['LotFrontage'])
plt.show()

############################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Lasso, LassoCV

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
test2=pd.read_csv("test.csv")
len_train=train.shape[0]
houses=pd.concat([train,test], sort=False)

# check type
houses.select_dtypes(include='object').head()

houses.select_dtypes(include=['float','int']).head()
#categorical
a =houses.select_dtypes(include='object').isnull().sum()[houses.select_dtypes(include='object').isnull().sum()>0]
print(a)
for col in ('Alley','Utilities','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
            'BsmtFinType2','Electrical','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond',
           'PoolQC','Fence','MiscFeature'):
    train[col]=train[col].fillna('None')
    test[col]=test[col].fillna('None')
#for col in ('MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):
    #train[col]=train[col].fillna(train[col].mode()[0])
    #test[col]=test[col].fillna(test[col].mode()[0])

for col in ('MSZoning','Exterior1st','Exterior2nd','KitchenQual','SaleType','Functional'):
    train[col]=train[col].fillna(train[col].mode()[0])
    test[col]=test[col].fillna(train[col].mode()[0])
#numerical
houses.select_dtypes(include=['int','float']).isnull().sum()[houses.select_dtypes(include=['int','float']).isnull().sum()>0]

for col in ('MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','GarageYrBlt','GarageCars','GarageArea'):
    train[col]=train[col].fillna(0)
    test[col]=test[col].fillna(0)
#train['LotFrontage']=train.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.mean()))
#test['LotFrontage']=test.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.mean()))

train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean())
test['LotFrontage']=test['LotFrontage'].fillna(train['LotFrontage'].mean())

print(train.isnull().sum().sum())
print(test.isnull().sum().sum())

#Remove some features high correlated and outliers
plt.figure(figsize=[30,15])
sns.heatmap(train.corr(), annot=True)

#from 2 features high correlated, removing the less correlated with SalePrice
train.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)
test.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)
#removing outliers recomended by author
train = train[train['GrLivArea']<4000]
len_train=train.shape[0]
print(train.shape)

houses=pd.concat([train,test], sort=False)

#transformation
houses['MSSubClass']=houses['MSSubClass'].astype(str)
#skew
skew=houses.select_dtypes(include=['int','float']).apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skew_df=pd.DataFrame({'Skew':skew})
skewed_df=skew_df[(skew_df['Skew']>0.5)|(skew_df['Skew']<-0.5)]
skewed_df.index

train=houses[:len_train]
test=houses[len_train:]

lam=0.1
for col in ('MiscVal', 'PoolArea', 'LotArea', 'LowQualFinSF', '3SsnPorch',
       'KitchenAbvGr', 'BsmtFinSF2', 'EnclosedPorch', 'ScreenPorch',
       'BsmtHalfBath', 'MasVnrArea', 'OpenPorchSF', 'WoodDeckSF',
       'LotFrontage', 'GrLivArea', 'BsmtFinSF1', 'BsmtUnfSF', 'Fireplaces',
       'HalfBath', 'TotalBsmtSF', 'BsmtFullBath', 'OverallCond', 'YearBuilt',
       'GarageYrBlt'):
    train[col]=boxcox1p(train[col],lam)
    test[col]=boxcox1p(test[col],lam)

train['SalePrice']=np.log(train['SalePrice'])

#Categorical to one hot encoding
houses=pd.concat([train,test], sort=False)
houses=pd.get_dummies(houses)

#prepare for model
train=houses[:len_train]
test=houses[len_train:]
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

x=train.drop('SalePrice', axis=1)
y=train['SalePrice']
test=test.drop('SalePrice', axis=1)
sc=RobustScaler()
x=sc.fit_transform(x)
test=sc.transform(test)

#model
model=Lasso(alpha =0.001, random_state=1)
model.fit(x,y)

#submission
pred=model.predict(test)
preds=np.exp(pred)
output=pd.DataFrame({'Id':test2.Id, 'SalePrice':preds})
output.to_csv('submission.csv', index=False)
output.head()

