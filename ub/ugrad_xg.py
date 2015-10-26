#modified public script version of UGrad file generator for the final ensemble
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import *
from sklearn import ensemble, tree
import xgboost as xgb
import sys
sys.path.append('../input')
import evaluation
from sklearn.ensemble import GradientBoostingClassifier
from hep_ml.uboost import uBoostClassifier
from hep_ml.gradientboosting import UGradientBoostingClassifier,LogLossFunction
from hep_ml.losses import BinFlatnessLossFunction, KnnFlatnessLossFunction

print("Load the training/test data using pandas")
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")
check_agreement = pd.read_csv('../input/check_agreement.csv')
check_correlation = pd.read_csv('../input/check_correlation.csv')

from feat import add_features

print("Adding features to both training and testing")
train = add_features(train)
test = add_features(test)

check_agreement = add_features(check_agreement)
check_correlation = add_features(check_correlation)

print("Eliminate SPDhits, which makes the agreement check fail")
from feat import filter_out

features = list(f for f in train.columns if f not in filter_out)

train_eval = train[train['min_ANNmuon'] > 0.4]

print("features:",features)
#train[features] = train[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#test[features] = test[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#check_agreement[features] = check_agreement[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#check_correlation[features] = check_correlation[features].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

print("Train a Random Forest model")
#rf1 = RandomForestClassifier(n_estimators=500, n_jobs=-1, criterion="entropy", max_depth=6, random_state=1)
#rf1 = RandomForestClassifier(n_estimators=500, n_jobs=-1, criterion="entropy", 
#      oob_score = True, class_weight = "subsample",
#max_depth=10, max_features=6, min_samples_leaf=2, random_state=1)
rf1 = RandomForestClassifier(n_estimators=600, n_jobs=4, criterion="entropy", 
      #oob_score = True, class_weight = "subsample",
      max_depth=None, max_features=9, min_samples_leaf=2, 
      min_samples_split=2, random_state=1)
rf1.fit(train[features], train["signal"])
#rf = ensemble.AdaBoostClassifier(n_estimators=50, learning_rate=0.098643,base_estimator=rf1)
#uniform_features  = ["mass"]
print("Train a UGradientBoostingClassifier")
loss = BinFlatnessLossFunction(['mass'], n_bins=15, uniform_label=0)
#loss = KnnFlatnessLossFunction(uniform_features, uniform_label=0)
rf = UGradientBoostingClassifier(loss=loss, n_estimators=500,  
                                  max_depth=6,
                                  #max_depth=7, min_samples_leaf=10,
                                  learning_rate=0.15, train_features=features, subsample=0.7, random_state=369)
rf.fit(train[features + ['mass']], train['signal'])

#loss_funct=LogLossFunction()
#rf=UGradientBoostingClassifier(loss=loss_funct,n_estimators=200, random_state=3,learning_rate=0.2,subsample=0.7)
#rf.fit(train[features],train["signal"])

print("Train a XGBoost model")
#params = {"objective": "binary:logistic",
#          "learning_rate": 0.2,
#          "max_depth": 6,
#          "min_child_weight": 3,
#          "silent": 1,
#          "subsample": 0.7,
#          "colsample_bytree": 0.7,
#          'nthread': 4,
#          "seed": 1}
#          
#num_trees=450

#don't use this xgboost output, use model selection instead!
params = {"objective": "binary:logistic",
          "learning_rate": 0.2,
          "max_depth": 8,
          'gamma': 0.01, # 0.005
          "min_child_weight": 5,
          "silent": 1,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          'nthread': 4,
          "seed": 1}

num_trees=600

#gbm = xgb.train(params, xgb.DMatrix(train[features], train["signal"]), num_trees)

agreement_probs= rf.predict_proba(check_agreement[features])[:,1]
print('Checking agreement...')
ks = evaluation.compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)
print ('KS metric UB =', ks, ks < 0.09)

train_eval_probs1 = rf.predict_proba(train_eval[features])[:,1]
AUC1 = evaluation.roc_auc_truncated(train_eval['signal'], train_eval_probs1)
print ('AUC UB ', AUC1)

print("Make predictions on the test set")
rfpred = rf.predict_proba(test[features])[:,1]
test_probs = rfpred
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
submission.to_csv("ub_only_submission.csv", index=False)
