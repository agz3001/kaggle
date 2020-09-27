# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 14:20:48 2019

@author: Shuhei
"""
"""
y =longitude
x =latitude
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
reader =pd.read_csv("train.csv", chunksize=50000)
#train =  pd.read_csv('../input/train.csv', nrows = 10_000_000)
test =pd.read_csv("test.csv")
train1 =pd.DataFrame(reader.get_chunk(50000))
#check type
print(train1.info())
print(train1.isnull().sum())
print(train1["fare_amount"].value_counts())
#transform
def manhattan(df):
    train1["abs_diff_longitude"] =(train1.dropoff_longitude-train1.pickup_longitude).abs()
    train1["abs_diff_latitude"] =(train1.dropoff_latitude-train1.pickup_latitude).abs()
def manhattan_test(df):
    test["abs_diff_longitude"] =(test.dropoff_longitude-test.pickup_longitude).abs()
    test["abs_diff_latitude"] =(test.dropoff_latitude-test.pickup_latitude).abs()

#データ多量、外れ値少量、この場合外れ値消去!!
#delete columns
train1 = train1.dropna(how = 'any', axis = 'rows')
manhattan(train1)
train1 = train1[(train1.abs_diff_longitude < 5.0) & (train1.abs_diff_latitude < 5.0)]
y_train =train1["fare_amount"]
x_train1 =train1.drop(["key","fare_amount","pickup_datetime","passenger_count", "pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude"],axis=1)

y_test =pd.read_csv("sample_submission.csv")
y_test =y_test["fare_amount"]
test =test.dropna(how = 'any', axis = 'rows')  #axisで指定した行か列に1つでもNaNが含まれる場合その行または列を削除。
manhattan_test(test)
test = test[(test.abs_diff_longitude < 5.0) & (test.abs_diff_latitude < 5.0)] #query??
x_test =test.drop(["key", "pickup_datetime","passenger_count","pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude"],axis=1)

#regression fit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x_train1, y_train, random_state=42)
lr =LinearRegression().fit(x_train1, y_train)
print(lr.score(x_train1, y_train))         #0.45496(n=10000), #0.46574(n=50000)
print(lr.score(x_test, y_test))            #-2.7712099353944373e+30

#regression svm(サポートベクタ回帰svR) n=10000以上だと処理が重いtext(p.102)
from sklearn import svm
reg = svm.SVR(kernel='rbf', C=10, gamma="auto").fit(x_train1, y_train)
print(reg.score(x_train1, y_train))        #0.6455067508150087(C=10), 0.6135161444083844(C=1)
print(reg.score(x_test, y_test))           #-4.67609903685942e+30

# 提出用ファイルの作成
submission = pd.DataFrame({'key': test['key'], 'fare_amount': reg.predict(x_test)})
submission.to_csv('submission1.csv', index=False)

import seaborn as sns
sns.distplot(train1['fare_amount'])
