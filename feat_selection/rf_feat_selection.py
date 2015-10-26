import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
import sys

from sklearn.cross_validation import *
from sklearn.grid_search import *

#import evaluation
from evaluation import *

import math

#from feat_traineval import add_features
#from feat_traineval import inv_feat_list

from feat_rf_good import add_features

print("Load the training/test data using pandas")

#train = pd.read_csv("../../input/feat_training_eval_only.csv")
#train = train[train['min_ANNmuon'] > 0.4]
train = pd.read_csv("../../input/feat_training.csv")
train_eval = train[train['min_ANNmuon'] > 0.4]

print("Eliminate SPDhits, which makes the agreement check fail")
#features = list(train.columns[1:-5])
#filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal', 
#              'SPDhits', 
#              #'p0_eta','p1_eta','p2_eta','LifeTime',
#              #'FlightDistanceError'
#              ]

#added_filterout = 'log_lifetime,p0p2_ip_ratio,p1p2_ip_ratio,3body_inv_mass,3body_trans_inv_mass,sum_dimuon_ip,p1p2_eta,p0p1_eta,p2p0_eta,pseudo_invmass12,pseudo_invmass02,pseudo_invmass01,pt0_tau_diff,pt1_tau_diff,pt2_tau_diff,distance_sec_vtx0,distance_sec_vtx1,distance_sec_vtx2,min_track_ip,max_track_ip,min_track_ipsig,max_track_ipsig,min_DCA,max_DCA,min_track_chi2,min_dimuon_ip,max_dimuon_ip,min_isolation_a_f,max_isolation_a_f,sum_isolation_a_f,min_isobdt,max_isobdt,min_CDF,max_CDF,sum_sqrt_track_chi2,sum_track_pt'.split(',')
added_filterout = 'log_lifetime,p0p2_ip_ratio,p1p2_ip_ratio,3body_inv_mass,3body_trans_inv_mass,sum_dimuon_ip,p1p2_eta,p0p1_eta,p2p0_eta,pseudo_invmass12,pseudo_invmass02,pseudo_invmass01,pt0_tau_diff,pt1_tau_diff,pt2_tau_diff,distance_sec_vtx0,distance_sec_vtx1,distance_sec_vtx2,min_track_ip,max_track_ip,min_track_ipsig,max_track_ipsig,min_DCA,max_DCA,min_track_chi2,min_dimuon_ip,max_dimuon_ip,min_isolation_a_f,max_isolation_a_f,sum_isolation_a_f,min_isobdt,max_isobdt,min_CDF,max_CDF,sum_sqrt_track_chi2,sum_track_pt'.split(',')
filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal',
              'p0_track_Chi2Dof','CDF1', 'CDF2', 'CDF3',
              #'isolationa', 'isolationb','isolationc', 'isolationd','isolatione', 'isolationf',
              'isolationb','isolationc', 
              'p0_pt','p1_pt', 'p2_pt', 'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta',
              'DOCAone', 'DOCAtwo', 'DOCAthree',
              'SPDhits'
              ]
filter_out+=added_filterout
#experiment_feat = ['min_isobdt','distance_sec_vtx1','p0p2_ip_ratio','p1p2_ip_ratio']
#experiment_feat = ['min_isobdt']
experiment_feat = ['p0p2_ip_ratio','p1p2_ip_ratio']

#filter_out +=inv_feat_list

#features = list(train.columns)
features = list(f for f in train.columns if f not in filter_out)
features+=experiment_feat
print features

print("Train a Random Forest model")

'''
parameters = {'n_estimators':[500],
              'n_jobs':[1],
              'criterion': ['gini', 'entropy'],
              'max_features': ['auto', 'sqrt'],
              'bootstrap': [True, False],
              'max_depth': [None, 5, 10, 20],
              'min_samples_split':[2, 5, 9],
              'min_samples_leaf': [1, 2, 5, 8]}
parameters = {'n_estimators':[600,1000],
              'n_jobs':[2],
              'criterion': ['entropy'],
              #'max_features': ['sqrt',6],
              'max_features': [8,9,10],
              'bootstrap': [True],
              'max_depth': [None,10,15],
              'min_samples_split':[2],
              'min_samples_leaf': [2]}
'''
parameters = {'n_estimators':[1000],
              'n_jobs':[2],
              'criterion': ['entropy'],
              #'max_features': ['sqrt',6],
              #'max_features': [10,None],
              'max_features': [10],
              'bootstrap': [True],
              'max_depth': [None],
              'min_samples_split':[2],
              'min_samples_leaf': [2]}

clf = RandomForestClassifier()

import evaluation

def _score_func(estimator, X, y):
    pred_probs = estimator.predict_proba(X)[:, 1]
    return evaluation.roc_auc_truncated(y, pred_probs)

clf = GridSearchCV(clf, parameters, n_jobs=1, cv=StratifiedKFold(train['signal'], n_folds=5, shuffle=True), 
                   scoring=_score_func,
                   verbose=3)

clf.fit(train[features], train['signal'])

for f, score in sorted(zip(features,clf.best_estimator_.feature_importances_), key=lambda x:x[1]):
    print '%s,%.5f'%(f,score)

print '-'*79
print 'RF best_params with new ip*dira remove all isolation'
print clf.best_score_
print clf.best_estimator_
print clf.best_estimator_.get_params()

print '-'* 79

'''
clf = ExtraTreesClassifier()

#clf = GridSearchCV(clf, parameters, n_jobs=5, cv=StratifiedKFold(train['signal'], n_folds=5, shuffle=True), 
#                   scoring=_score_func,
#                   verbose=3)
clf = GridSearchCV(clf, parameters, n_jobs=5, cv=StratifiedKFold(train_eval['signal'], n_folds=5, shuffle=True), 
                   scoring=_score_func,
                   verbose=3)

clf.fit(train[features], train['signal'])

print 'ExtraTrees best_params'
print clf.best_score_
print clf.best_estimator_
print clf.best_estimator_.get_params()
#print features
#print clf.best_estimator_.feature_importances_
#for f, score in sorted(zip(features,clf.best_estimator_.feature_importances_), key=lambda x:x[1]):
#    print '%s,%.5f'%(f,score)
'''
test_agreement_ = True
if test_agreement_:
    print 'test agreement, relax...'
    check_agreement = pd.read_csv('../../input/feat_check_agreement.csv')
    check_correlation = pd.read_csv('../../input/feat_check_correlation.csv')
    #check_agreement = pd.read_csv('../input/check_agreement.csv')
    #check_correlation = pd.read_csv('../input/check_correlation.csv')
    #check_agreement = add_features(check_agreement)
    #check_correlation = add_features(check_correlation)

    agreement_probs= (clf.predict_proba(check_agreement[features])[:,1])

    ks = compute_ks(
            agreement_probs[check_agreement['signal'].values == 0],
            agreement_probs[check_agreement['signal'].values == 1],
            check_agreement[check_agreement['signal'] == 0]['weight'].values,
            check_agreement[check_agreement['signal'] == 1]['weight'].values)
    print ('KS metric', ks, ks < 0.09)

    correlation_probs = clf.predict_proba(check_correlation[features])[:,1]
    print ('Checking correlation...')
    cvm = compute_cvm(correlation_probs, check_correlation['mass'])
    print ('CvM metric', cvm, cvm < 0.002)

    train_eval_probs = clf.predict_proba(train_eval[features])[:,1]
    print ('Calculating AUC...')
    AUC = roc_auc_truncated(train_eval['signal'], train_eval_probs)
    print ('AUC', AUC)

gen_test_extra = True
if gen_test_extra:
    test  = pd.read_csv("../../input/feat_test.csv")
    test_probs = clf.best_estimator_.predict_proba(test[features])[:,1]
    submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
    submission.to_csv("randomforest_withp1p2ip_low_score_cv_%.5f_ks_%.5f_auc_%.10f.csv"%(clf.best_score_, ks,AUC), index=False)
    #test_probs = clf.predict_proba(test[features])[:,1]
    #submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
    #submission.to_csv("randomforest_low_score_cv_%.5f_auc_%.10f_bk.csv"%(clf.best_score_, AUC), index=False)
