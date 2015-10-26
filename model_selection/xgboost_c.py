#xgboost wrapper for parameter search
import inspect
import os
import sys
#code_path = os.path.join(
#	os.path.split(inspect.getfile(inspect.currentframe()))[0], "xgboost/wrapper")
#sys.path.append(code_path)
import xgboost as xgb
import numpy as np

class XGBC(object):
	def __init__(self, num_round = 100, max_depth = 5, eta= 0.1, min_child_weight = 2, 
		colsample_bytree = 1, gamma = 0, subsample = 1, seed = 1337):
		self.max_depth = max_depth
		self.eta = eta
		self.colsample_bytree = colsample_bytree
		self.num_round = num_round
		self.min_child_weight = min_child_weight
		self.gamma = gamma
		self.subsample = subsample
		self.seed = seed
	def fit(self, train, label):
		dtrain = xgb.DMatrix(train, label = label, missing = np.nan)
		param = {'max_depth':self.max_depth, 'eta':self.eta, 'silent':1, 'colsample_bytree': self.colsample_bytree, 
		'min_child_weight': self.min_child_weight, 'objective':'binary:logistic', 'gamma':self.gamma, 
		'subsample': self.subsample, 'seed':self.seed}
		self.bst = xgb.train(param, dtrain, self.num_round)
	def predict_proba(self, test):
		dtest = xgb.DMatrix(test, missing = np.nan)
		ypred = self.bst.predict(dtest)
		return np.column_stack((1-ypred, ypred))
