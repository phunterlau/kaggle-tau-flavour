import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import sys

import evaluation
from evaluation import *

from sklearn.cross_validation import *
from sklearn.grid_search import GridSearchCV
import math

""" Implemented Scikit- style grid search to find optimal XGBoost params"""
""" Use this module to identify optimal hyperparameters for XGBoost"""

print("Load the training/test data using pandas")

train = pd.read_csv("../../input/feat_training.csv")
train_eval = train[train['min_ANNmuon'] > 0.4]

print('finish adding features')

added_filterout = 'log_lifetime,p0p2_ip_ratio,p1p2_ip_ratio,3body_inv_mass,3body_trans_inv_mass,sum_dimuon_ip,p1p2_eta,p0p1_eta,p2p0_eta,pseudo_invmass12,pseudo_invmass02,pseudo_invmass01,pt0_tau_diff,pt1_tau_diff,pt2_tau_diff,distance_sec_vtx0,distance_sec_vtx1,distance_sec_vtx2,min_track_ip,max_track_ip,min_track_ipsig,max_track_ipsig,min_DCA,max_DCA,min_track_chi2,min_dimuon_ip,max_dimuon_ip,min_isolation_a_f,max_isolation_a_f,sum_isolation_a_f,min_isobdt,max_isobdt,min_CDF,max_CDF,sum_sqrt_track_chi2,sum_track_pt'.split(',')
filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal',
              'p0_track_Chi2Dof','CDF1', 'CDF2', 'CDF3',
              'isolationb', 'isolationc','p0_pt',
              'p1_pt', 'p2_pt', 'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta',
              'DOCAone', 'DOCAtwo', 'DOCAthree',
              'SPDhits'
              ]
filter_out+=added_filterout

features = list(f for f in train_eval.columns if f not in filter_out)
experiment_feat = ['p0p2_ip_ratio','p1p2_ip_ratio']
features+=experiment_feat
print features

print("Train a XGBoost model")

class XGBoostClassifier():

    def __init__(self, num_boost_round=600, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'binary:logistic'})

    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        self.label2num = {label: i for i, label in enumerate(sorted(set(y)))}
        dtrain = xgb.DMatrix(X, label=[self.label2num[label] for label in y])
        self.clf = xgb.train(self.params, dtrain, num_boost_round)

    def predict(self, X):
        num2label = {i: label for label, i in self.label2num.items()}
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return np.array([num2label[i] for i in y])

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        return self.clf.predict(dtest)

    def score(self, X, y):
        Y = self.predict_proba(X)
        return evaluation.roc_auc_truncated(y, Y[:, 1])

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **params):
        if 'num_boost_round' in params:
            self.num_boost_round = params.pop('num_boost_round')
        if 'objective' in params:
            del params['objective']
        self.params.update(params)
        return self

clf = XGBoostClassifier(
        eval_metric='auc',
        objective='binary:logistic',
        num_class=2,
        nthread=4,
        silent=1,
)

#put the parameter list as standalone for consistent experiment record, 
#e.g. para_list1.py: AUC=0.98xx para_list2.py: AUC=0.98xx
#from para_list_feat_selection import parameters #not working, only 0.91657984120945146
from para_list_feat_selection import parameters

# preserves label percentages across each splitting/ shuffling and fitting step

#clf = GridSearchCV(clf, parameters, n_jobs=4, cv=StratifiedKFold(train['signal'], n_folds=5, shuffle=True), 
#        verbose=3)
#clf = GridSearchCV(clf, parameters, n_jobs=5, cv=StratifiedKFold(train['signal'], n_folds=5, shuffle=True), 
#        refit=True,
#        verbose=3)
clf = GridSearchCV(clf, parameters, n_jobs=5, cv=StratifiedKFold(train_eval['signal'], n_folds=5, shuffle=True), 
        refit=True,
        verbose=3)

clf.fit(train[features], train["signal"])

best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('score:', score)
for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

print 'xgboost best_params'
print clf.best_score_
print clf.best_estimator_
print clf.best_estimator_.get_params()

test_agreement_ = True
if test_agreement_:
    print 'test agreement, relax...'
    check_agreement = pd.read_csv('../../input/feat_check_agreement.csv')
    check_correlation = pd.read_csv('../../input/feat_check_correlation.csv')

    #agreement_probs= (clf.predict_proba(check_agreement[features])[:,1])
    agreement_probs= (clf.best_estimator_.predict_proba(check_agreement[features])[:,1])

    ks = compute_ks(
            agreement_probs[check_agreement['signal'].values == 0],
            agreement_probs[check_agreement['signal'].values == 1],
            check_agreement[check_agreement['signal'] == 0]['weight'].values,
            check_agreement[check_agreement['signal'] == 1]['weight'].values)
    print ('KS metric', ks, ks < 0.09)

    correlation_probs = clf.best_estimator_.predict_proba(check_correlation[features])[:,1]
    print ('Checking correlation...')
    cvm = compute_cvm(correlation_probs, check_correlation['mass'])
    print ('CvM metric', cvm, cvm < 0.002)

    train_eval_probs = clf.best_estimator_.predict_proba(train_eval[features])[:,1]
    print ('Calculating AUC...')
    AUC = roc_auc_truncated(train_eval['signal'], train_eval_probs)
    print ('AUC', AUC)

gen_test_xg = True
if gen_test_xg:
    test  = pd.read_csv("../../input/feat_test.csv")
    test_probs = clf.best_estimator_.predict_proba(test[features])[:,1]
    submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
    submission.to_csv("xg_low_score_cv_%.5f.csv"%(clf.best_score_), index=False)
