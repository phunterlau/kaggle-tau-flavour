'''
create some high score not passing predictions
'''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF
import xgboost as xgb
import evaluation
import math
from xgboost_c import XGBC
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import roc_auc_score as AUC
import random

def cv_model_list():
	model_list = []
	for num_round in [200]:
		for max_depth in [5,10,15]:
			for eta in [0.1]:
				for min_child_weight in [5]:
					for colsample_bytree in [0.75,1]:
						for subsample in [0.75,1]:
							model_list.append((XGBC(num_round = num_round, max_depth = max_depth, eta = eta, min_child_weight = min_child_weight,
								colsample_bytree = colsample_bytree, subsample = subsample), 
							'xgb_tree_%i_depth_%i_lr_%f_child_%i'%(num_round, max_depth, eta, min_child_weight)))
	
	return model_list

from feat import add_features

def gen_data():
	path = '../data/'
	print "loading data..."
	train = pd.read_csv(path + "training.csv")
	test  = pd.read_csv(path + "test.csv")
	train, test = add_features(train), add_features(test)

	return train, test

def delete_features(df):
	'''
	filter_out = ['id','signal','min_ANNmuon', 'production', 'mass', 
	'SPDhits', 'p0_eta','p1_eta','p2_eta','LifeTime','FlightDistanceError','weight']
	'''
	#filter_out = ['id','signal','min_ANNmuon', 'production', 'mass', 'weight']
    from feat import filter_out
    filter_out.remove('SPDhits') #use SPD hits
	#features = list(train.columns)
	features = list(f for f in df.columns if f not in filter_out)
	return df[features]


def cv_model(model_list):
	print "generating cv csv files...."
	train, test = gen_data()
	label = train['signal']
	train_id = train.id
	test_id = test.id

	train_del, test_del = delete_features(train), delete_features(test)

	check_agreement = pd.read_csv('../data/check_agreement.csv')
	check_correlation = pd.read_csv('../data/check_correlation.csv')
	check_agreement= add_features(check_agreement)
	check_correlation  = add_features(check_correlation)

	X, X_test = train_del.as_matrix(), test_del.as_matrix()
	print X.shape, X_test.shape

	kf = KFold(label, n_folds = 4)
	for j, (clf, clf_name) in enumerate(model_list):
		
		print "modelling %s...."%clf_name
		cv_train = np.zeros(len(label))
		for i, (train_fold, validate) in enumerate(kf):
			X_train, X_validate, label_train, label_validate = X[train_fold,:], X[validate,:], label[train_fold], label[validate]
			clf.fit(X_train,label_train)
			cv_train[validate] = clf.predict_proba(X_validate)[:,1]
		print "the true roc_auc_truncated is %.6f"%evaluation.roc_auc_truncated(label[train['min_ANNmuon'] > 0.4], 
			pd.Series(cv_train)[train['min_ANNmuon'] > 0.4])
		# save the cv
		cv_sub = pd.DataFrame({"id": train_id, "prediction": cv_train})
		cv_sub.to_csv("../data/high_score/cv/xgb%i.csv"%j, index=False)
		# save the prediction
		clf.fit(X, label)
		test_probs = clf.predict_proba(X_test)[:,1]
		submission = pd.DataFrame({"id": test_id, "prediction": test_probs})
		submission.to_csv("../data/high_score/pred/xgb%i.csv"%j, index=False)
		# check if it passes the tests
		print "check if it passes the tests"
		agreement_probs = clf.predict_proba(delete_features(check_agreement).as_matrix())[:,1]
		ks = evaluation.compute_ks(
			agreement_probs[check_agreement['signal'].values == 0],
			agreement_probs[check_agreement['signal'].values == 1],
			check_agreement[check_agreement['signal'] == 0]['weight'].values,
			check_agreement[check_agreement['signal'] == 1]['weight'].values)
		print ('KS metric', ks, ks <= 0.09)
		# save agreement
		submission = pd.DataFrame({"id": check_agreement['id'], "prediction": agreement_probs})
		submission.to_csv("../data/high_score/agreement/xgb%i.csv"%j, index=False)

		correlation_probs = clf.predict_proba(delete_features(check_correlation).as_matrix())[:,1]
		print ('Checking correlation...')
		cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])
		print ('CvM metric', cvm, cvm <= 0.002)
		# save correlation
		submission = pd.DataFrame({"id": check_correlation['id'], "prediction": correlation_probs})
		submission.to_csv("../data/high_score/correlation/xgb%i.csv"%j, index=False)

if __name__ == '__main__':
	cv_model(cv_model_list())
	print "ALL DONE!"



