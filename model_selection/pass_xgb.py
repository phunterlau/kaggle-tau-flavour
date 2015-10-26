'''
create some high score not passing predictions
'''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LGR
import xgboost as xgb
import evaluation
import math
from xgboost_c import XGBC
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import roc_auc_score as AUC
import random

def cv_model_list():
	model_list = []
	for num_round in [700]:
		for max_depth in [8]:
			for eta in [0.1,0.2]:
				for min_child_weight in [10,15,25]:
					for colsample_bytree in [0.25,0.35,0.45,0.6]:
						for subsample in [0.6,0.7]:
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
	#filter_out = ['id','signal','min_ANNmuon', 'production', 'mass', 
	#'SPDhits', 'p0_eta','p1_eta','p2_eta','LifeTime','FlightDistanceError','weight']
    from feat import filter_out
	#filter_out = ['id','signal','min_ANNmuon', 'production', 'mass', 'weight']
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
		
		print "modelling model %i ...."%j
		cv_train = np.zeros(len(label))
		for i, (train_fold, validate) in enumerate(kf):
			X_train, X_validate, label_train, label_validate = X[train_fold,:], X[validate,:], label[train_fold], label[validate]
			clf.fit(X_train,label_train)
			cv_train[validate] = clf.predict_proba(X_validate)[:,1]
		auc_score = evaluation.roc_auc_truncated(label[train['min_ANNmuon'] > 0.4], 
			pd.Series(cv_train)[train['min_ANNmuon'] > 0.4])
		print "the true roc_auc_truncated is %.6f"%auc_score

		clf.fit(X, label)
		test_probs = clf.predict_proba(X_test)[:,1]
		# check if it passes the tests
		print "check if it passes the tests"
		agreement_probs = clf.predict_proba(delete_features(check_agreement).as_matrix())[:,1]
		ks = evaluation.compute_ks(
			agreement_probs[check_agreement['signal'].values == 0],
			agreement_probs[check_agreement['signal'].values == 1],
			check_agreement[check_agreement['signal'] == 0]['weight'].values,
			check_agreement[check_agreement['signal'] == 1]['weight'].values)
		print ('KS metric', ks, ks <= 0.09)

		correlation_probs = clf.predict_proba(delete_features(check_correlation).as_matrix())[:,1]
		print ('Checking correlation...')
		cvm = evaluation.compute_cvm(correlation_probs, check_correlation['mass'])
		print ('CvM metric', cvm, cvm <= 0.002)
		#if ks <= 0.09 and cvm <= 0.002 and auc_score > 0.975: # no need to check here
		if auc_score > 0.965: # the minimum threshold
			# save the cv
			cv_sub = pd.DataFrame({"id": train_id, "prediction": cv_train, "label": label})
			cv_sub.to_csv("../data/cv_folder/xgb%i.csv"%j, index=False)
			# save the prediction
			submission = pd.DataFrame({"id": test_id, "prediction": test_probs})
			submission.to_csv("../data/pred_folder/xgb%i.csv"%j, index=False)
			# save agreement
			submission = pd.DataFrame({"id": check_agreement['id'], "prediction": agreement_probs})
			submission.to_csv("../data/agree_folder/xgb%i.csv"%j, index=False)
			# save correlation
			submission = pd.DataFrame({"id": check_correlation['id'], "prediction": correlation_probs})
			submission.to_csv("../data/correlation_folder/xgb%i.csv"%j, index=False)

if __name__ == '__main__':
	cv_model(cv_model_list())
	print "ALL DONE!"



