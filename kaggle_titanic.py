# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 01:58:24 2019

@author: Shuhei
"""
import numpy as np
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

x_train = train.drop(['Survived'], axis=1)  # 学習データを特徴量と目的変数に分ける
y_train = train['Survived']
x_test = test.copy() # テストデータは特徴量のみなので、そのまま
y_test =pd.read_csv("gender_submission.csv")
y_test =y_test["Survived"]
print(train.Sex.value_counts())
# 特徴量作成
from sklearn.preprocessing import LabelEncoder

x_train = x_train.drop(['PassengerId'], axis=1)  # PassengerIdを除外する
x_test = x_test.drop(['PassengerId'], axis=1)
x_train = x_train.drop(['Name', 'Ticket', 'Cabin'], axis=1)  # Name, Ticket, Cabinを除外する
x_test = x_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# それぞれのカテゴリ変数にlabel encodingを適用する.ダミー変換ではなく、数値を割り当てるラベル変換!
#欠損値があれば nanで置き換え. name,ticket,cabinは削除したから,ここでは残るsex, embarkedだけ
for c in ['Sex', 'Embarked']:
    # 学習データに基づいてどう変換するかを定める
    le = LabelEncoder()
    le.fit(x_train[c].fillna('NA'))

    # 学習データ、テストデータを変換する
    x_train[c] = le.transform(x_train[c].fillna('NA'))
    x_test[c] = le.transform(x_test[c].fillna('NA'))

# 数値変数の欠損値を学習データの平均で埋める
num_cols = ['Age', 'SibSp', 'Parch', 'Fare']
for col in num_cols:
    x_train[col].fillna(x_train[col].mean(), inplace=True)
    x_test[col].fillna(x_train[col].mean(), inplace=True)

# 変数Fareを対数変換する
x_train['Fare'] = np.log1p(x_train['Fare'])
x_test['Fare'] = np.log1p(x_test['Fare'])

########################################################

from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgbm

#data for...
kfold =KFold(n_splits=4, shuffle =True, random_state=70) #trainを4分割しそのうち1つをvalにつかう
train_idx, val_idx =list(kfold.split(x_train))[0]
x_train, x_val =x_train.loc[train_idx], x_train.loc[val_idx]
y_train, y_val =y_train.loc[train_idx], y_train.loc[val_idx]
#for xgb
dtrain =xgb.DMatrix(x_train, label=y_train) #label:ターゲット(答え)のこと
dtest =xgb.DMatrix(x_test)#テストデータが行列でも対応可
dvalid =xgb.DMatrix(x_val, label=y_val)
#for lgbm
lgbm_train =lgbm.Dataset(x_train, y_train, free_raw_data=False) #lgbmはDMatrix使えないらしいからDatasetで
lgbm_val =lgbm.Dataset(x_val, y_val)


#objective(目的関数):回帰は"reg:squarederror",2値分類は"binary:logistic",他クラス分類は"multi:softprob"
param = {'booster': 'dart',#gblinear:線形の場合
         'objective': 'binary:logistic',
         "silent":1, #silent=0 or verbosity=1.printing messages.表示か非表示か
         "random_state":70,
         'max_depth': 5, #
         'learning_rate': 0.1,
         'sample_type': 'uniform',#"weighted": dropped_trees selected in proportion to weight."unform":selected uniformly.
         'normalize_type': 'tree',#or forest.Weight of new trees are 1 / (1 + learning_rate)
         'rate_drop': 0.1,#drop_out rate
         'skip_drop': 0.5,#default=0, non-zero skip_drop has higher priority than rate_drop or one_drop
         "eval_metric":"logloss"}#"eval_metric":オプション
num_round = 50 #作成する決定木の数
model = xgb.train(param, dtrain, num_round)
model.predict(dtest, ntree_limit=num_round) # ntree_limit not be 0
xgb.plot_importance(model)#xgboostクラス分類での特徴量の重要度マップ(×回帰)
#validation on xgboost
#modelの [early_stopping_rounds],predictの[ntree_limit]は早期停止のオプション
watchlist =[(dtrain,"train"), (dvalid,"eval")]
model = xgb.train(param, dtrain, num_round, evals=watchlist)
val_pred =model.predict(dvalid)
score =log_loss(y_val, val_pred)
print("logloss:{:.4f}".format(score))
pred =model.predict(dtest)
y_pred =np.where(pred >0.5, 1,0)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print("test accuracy:", acc)

