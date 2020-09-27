# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import os
import mglearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgbm

#data,rings=28なし！！
abalone_path =os.path.join(mglearn.datasets.DATA_PATH, "abalone.data")
data3 =pd.read_csv(abalone_path, header=None, index_col=False,
                   names=["sex","length","diameter","height","whole_weight",
                          "shucked_weight","viscera_weight","shell_weight","rings"])
for c in ["sex"]:
    le =LabelEncoder()
    data3[c] =le.fit_transform(data3[c])
y =data3.loc[:,"rings"]
x =data3.drop(["rings"], axis=1)

sns.distplot(y)
x_train, x_test, y_train,y_test =train_test_split(x,y, random_state=0)
#data for validation
kfold =KFold(n_splits=4, shuffle =True, random_state=71) #trainを4分割しそのうち1つをvalにつかう
train_idx, val_idx =list(kfold.split(x_train))[0]
x_train, x_val =x_train.iloc[train_idx], x_train.iloc[val_idx]
y_train, y_val =y_train.iloc[train_idx], y_train.iloc[val_idx]
#for xgb
dtrain =xgb.DMatrix(x_train, label=y_train) #label:ターゲット(答え)のこと
dtest =xgb.DMatrix(x_test)#テストデータが行列でも対応可
dvalid =xgb.DMatrix(x_val, label=y_val)
#scikit-learn,XGBoost
model =XGBClassifier(n_estimator=20, random_state=71)
model.fit(x_train, y_train)
pred =model.predict(x_test)
xgb.plot_importance(model)
print("xgboost accuracy on train set:{}".format(model.score(x_train, y_train)))
print("xgboost accuracy on test set:{}".format(model.score(x_test, y_test)))
print("confusion matrix:\n{}".format(confusion_matrix(y_test, pred)))
print("classification report:\n{}".format(classification_report(y_test, pred)))


pred1 =(model.predict_proba(x_test)[:,1]>0.1).astype(bool)
pred2 =model.predict_proba(x_test)
best_preds = np.asarray([np.argmax(line) for line in pred2])

#
#xgboost
#objective(目的関数):回帰は"reg:squarederror",2値分類は"binary:logistic",他クラス分類は"multi:softprob"
param = {'booster': 'dart',#gblinear:線形の場合
         'objective': 'multi:softprob',#"mult:softprob"の場合、"metric":"mlogloss"に変更
         "silent":1, #silent=0 or verbosity=1.printing messages.表示か非表示か
         "random_state":70,#"seed":None(default):あまり使われない.これを固定すると結果が再現されやすい。これ調整したスコアの変化みれば、スコアの変化がランダムによるものなのか、パラメータ変更による改善なのか推測できる
         'max_depth': 5, #default=6
         'learning_rate': 0.1,
         'sample_type': 'uniform',#"weighted": dropped_trees selected in proportion to weight."unform":selected uniformly.
         'normalize_type': 'tree',#or forest.Weight of new trees are 1 / (1 + learning_rate)
         'rate_drop': 0,#drop_out rate
         'skip_drop': 0.5,#default=0, non-zero skip_drop has higher priority than rate_drop or one_drop
         "eval_metric":"mlogloss",#"eval_metric":オプション
         "num_class":28}
num_round = 50 #作成する決定木の数
model = xgb.train(param, dtrain, num_round)
#validation on xgboost
#modelの [early_stopping_rounds],predictの[ntree_limit]は早期停止のオプション
watchlist =[(dtrain,"train"), (dvalid,"eval")]
model = xgb.train(param, dtrain, num_round, evals=watchlist)
#損失関数の検証
#図描く必要ないならこれ.その場合modelのevals_result={}消す
val_pred =model.predict(dvalid)
score =log_loss(y_val, val_pred) #validationのスコア
print("logloss:{:.4f}".format(score))

#accuracy score
y_pred_proba = model.predict(dtest,ntree_limit=num_round)#提出用の予測値算出
best_preds = np.asarray([np.argmax(line) for line in y_pred_proba])
acc = accuracy_score(y_test, best_preds) #np.mean(y_test==best_preds)でも可
print("test accuracy:", acc)

#-----------------------------------------------------------
#parameter tuning
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import log_loss

# xgboostによる学習・予測を行うクラス
class Model:

    def __init__(self, params=None):
        self.model = None
        if params is None:
            self.params = {}
        else:
            self.params = params

    def fit(self, x_train, y_train, x_val, y_val):
        params = {'objective': "multi:softprob", 'silent': 1, 'random_state': 71,"num_class":29}
        params.update(self.params)
        num_round = 10
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dvalid = xgb.DMatrix(x_val, label=y_val)
        watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
        self.model = xgb.train(params, dtrain, num_round, evals=watchlist)

    def predict(self, x):
        data = xgb.DMatrix(x)
        pred = self.model.predict(data)
        return pred
# hyperoptを使ったパラメータ探索
def score(params):
    # パラメータを与えたときに最小化する評価指標を指定する
    # 具体的には、モデルにパラメータを指定して学習・予測させた場合のスコアを返すようにする
    params['max_depth'] = int(params['max_depth']) # max_depthの型を整数型に修正する
    model = Model(params)
    model.fit(x_train, y_train, x_val, y_val)
    val_pred = model.predict(x_val)
    score = log_loss(y_val, val_pred)
    print(f'params: {params}, logloss: {score:.4f}')
    history.append((params, score))
    return {'loss': score, 'status': STATUS_OK}
# 探索するパラメータ空間.p312(kaggle-master)
params = {
    'booster': 'gbtree',
    'objective': 'multi:softprob',
    'eta': 0.1,#チューニングでは固定,モデル作成では小さくする.計算量に影響する.量が多いと0.05まで下げてもいい
    'gamma': 0.0,#[1e-8,1.0]対数が一様分布に従う
    'alpha': 0.0,#default値.余裕あれば調整#1e-5,1e-2,0.1,1,100の順序で
    'lambda': 1.0,#default値.余裕あれば調整
    'min_child_weight': 1,#[0.1,10.0]対数が一様分布に従う
    'max_depth': 5,#[3,9]1刻み
    'subsample': 0.8,#[o.6,0.95]0.05刻み
    'colsample_bytree': 0.8,#[0.6,0.95]0.05刻み.colsample_bylevel:決定木ではなく分岐の深さでやる
    'random_state': 71,#乱数
    "num_class":29
}
num_round=1000#early_stopping_roundで調整.early_stopping =10/eta
# パラメータの探索範囲
param_space = {
    'min_child_weight': hp.loguniform('min_child_weight', np.log(0.1), np.log(10)),
    'max_depth': hp.quniform('max_depth', 3, 9, 1),#最も重要
    'subsample': hp.quniform('subsample', 0.6, 0.95, 0.05),#0.6-1.0まで試す
    'colsample_bytree': hp.quniform('colsample_bytree', 0.6, 0.95, 0.05),#0.6-1.0まで試す
    'gamma': hp.loguniform('gamma', np.log(1e-8), np.log(1.0)),#0.0-0.4まで試す
}
# hyperoptによるパラメータ探索の実行
max_evals = 10
trials = Trials()
history = []
fmin(score, param_space, algo=tpe.suggest, trials=trials, max_evals=max_evals)

# 記録した情報からパラメータとスコアを出力する
# （trialsからも情報が取得できるが、パラメータの取得がやや行いづらいため）
history = sorted(history, key=lambda tpl: tpl[1])
best = history[0]
print(f'best params:{best[0]}, score:{best[1]:.4f}')

#特徴量選択
#xgboost
dtrain =xgb.DMatrix(x_train, y_train)
params ={"objective":'multi:softprob',"silent":"1","random_state":71}
num_round =50
model =xgb.train(params, dtrain, num_round)
fscore =model.get_score(importance_type="total_gain")

xgb.plot_importance(model)#xgboostクラス分類での特徴量の重要度マップ(×回帰)
xgb.plot_tree(model, num_trees=2)


#------------------------------------------------------------------
ringsのパレート図

data4 =pd.DataFrame(data3.rings.value_counts())
data4["percent"] =data4["rings"]/sum(data4["rings"])*100
data4["accum"] =np.cumsum(data4["percent"])
data4 =data4.sort_values(by="rings",ascending=False)

fig, ax=plt.subplots()

ax.bar(range(len(data4.rings)),data4.rings,tick_label=data4.index)
ax.set_xlabel("rings")
ax.set_ylabel("count")

ax1 =ax.twinx()
ax1.plot(range(len(data4.rings)), data4.accum,color="red", linestyle="-.")
ax1.hlines(y=80,xmin=0, xmax=29)
ax1.set_title("pareto_chart")
plt.show()


