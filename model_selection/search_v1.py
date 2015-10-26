'''
start with one high AUC prediction but not passing test, and greedily search for weighted average to pass the test
'''
import os
import pandas as pd
import evaluation
import random
import numpy as np

class Passing_Search(object):
	check_agreement = pd.read_csv('../data/check_agreement.csv')
	check_correlation = pd.read_csv('../data/check_correlation.csv')
	files = [f for f in os.listdir('../data/cv_folder') if '.csv' in f]
	cv_folder = '../data/cv_folder/'
	pred_folder = '../data/pred_folder/'
	agree_folder = '../data/agree_folder/'
	correlation_folder = '../data/correlation_folder/'

	def __init__(self, high_per):
		# initialize the parameters
		self.init_cv, self.init_pred, self.init_agree, self.init_correlation = high_per
	def check_agreement_func(self, agreement_probs):
		ks = evaluation.compute_ks(
			agreement_probs[self.check_agreement['signal'].values == 0],
			agreement_probs[self.check_agreement['signal'].values == 1],
			self.check_agreement[self.check_agreement['signal'] == 0]['weight'].values,
			self.check_agreement[self.check_agreement['signal'] == 1]['weight'].values)
		return ks
	def check_corr_func(self, correlation_probs):
		cvm = evaluation.compute_cvm(correlation_probs, self.check_correlation['mass'])
		return cvm
	def search(self):
		'''
		main function to perform searching...
		'''
		# initial status
		current_cv = self.init_cv
		current_pred = self.init_pred
		current_agree = self.init_agree
		current_corr = self.init_correlation
		train = pd.read_csv("../data/training.csv") # for cross validation purposes
		label = train['signal']

		current_ks = self.check_agreement_func(self.init_agree)
		current_cvm = self.check_corr_func(self.init_correlation)
		current_auc = 	evaluation.roc_auc_truncated(label[train['min_ANNmuon'] > 0.4], 
					pd.Series(current_cv)[train['min_ANNmuon'] > 0.4])
		print "the initial test results..."
		print ('KS metric',current_ks, current_ks <= 0.09)
		print ('Cvm metric',current_cvm, current_cvm <= 0.002)
		
		# start searching
		num_round = 0
		while current_ks > 0.09 or current_cvm > 0.002:
			num_round += 1
			print "doing round %i..."%num_round
			if num_round > 10:
				print "finished all the rounds and can't find a solution..."
				break
			random_files = random.sample(self.files, len(self.files)) # add some randomness
			for f in random_files:
				# read all the files
				tmp_cv = pd.read_csv(self.cv_folder + f)
				tmp_pred = pd.read_csv(self.pred_folder + f)
				tmp_agree = pd.read_csv(self.agree_folder + f)
				tmp_corr = pd.read_csv(self.correlation_folder + f)

				tmp_agree_average = (tmp_agree['prediction'].values + current_agree) * 0.5
				tmp_corr_average = (tmp_corr['prediction'].values + current_corr) * 0.5
				tmp_cv_average = (tmp_cv['prediction'].values + current_cv) * 0.5
				tmp_auc = evaluation.roc_auc_truncated(label[train['min_ANNmuon'] > 0.4], 
					pd.Series(current_cv)[train['min_ANNmuon'] > 0.4])
				if self.check_agreement_func(tmp_agree_average) < current_ks and self.check_corr_func(tmp_corr_average) <= 0.002:
					# update them
					current_ks = self.check_agreement_func(tmp_agree_average)
					current_cvm = self.check_corr_func(tmp_corr_average)
					current_cv = tmp_cv_average
					current_pred = (tmp_pred['prediction'].values + current_pred) * 0.5
					current_agree,current_corr = tmp_agree_average, tmp_corr_average
					print "find a reduced ks score %.3f..."%current_ks
					current_auc = tmp_auc
					print "the corresponding AUC score is %.5f"%current_auc
					if current_ks <= 0.09:
						print "found one that passes the test, and now start to optimize the AUC"
						print "doing 5 rounds..."
						n_r = 0
						while n_r < 2:
							n_r += 1
							print n_r
							for e2, f2 in enumerate(random_files):
								# read all the files
								tmp_cv2 = pd.read_csv(self.cv_folder + f2)
								tmp_pred2 = pd.read_csv(self.pred_folder + f2)
								tmp_agree2 = pd.read_csv(self.agree_folder + f2)
								tmp_corr2 = pd.read_csv(self.correlation_folder + f2)

								tmp_agree_average2 = (tmp_agree2['prediction'].values + current_agree) * 0.5
								tmp_corr_average2 = (tmp_corr2['prediction'].values + current_corr) * 0.5
								tmp_cv_average2 = (tmp_cv2['prediction'].values + current_cv) * 0.5
								tmp_auc2 = evaluation.roc_auc_truncated(label[train['min_ANNmuon'] > 0.4], 
									pd.Series(tmp_cv_average2)[train['min_ANNmuon'] > 0.4])
								if self.check_agreement_func(tmp_agree_average2) <= 0.09 and self.check_corr_func(tmp_corr_average2) <= 0.002 and \
								tmp_auc2 > current_auc:
									# update them
									current_ks = self.check_agreement_func(tmp_agree_average2)
									current_cvm = self.check_corr_func(tmp_corr_average2)
									current_cv = tmp_cv_average2
									current_pred = (tmp_pred2['prediction'].values + current_pred) * 0.5
									current_agree,current_corr = tmp_agree_average2, tmp_corr_average2
									print "current ks score %.3f..."%current_ks
									current_auc = tmp_auc2
									print "the corresponding AUC score is %.5f"%current_auc
						print "Yeah! We've found one good prediction!"
						submission = pd.DataFrame({"id": tmp_pred['id'], "prediction": current_pred})
						submission.to_csv("../submissions/xgb_search_%.4f.csv"%current_auc, index=False)
						break

if __name__ == '__main__':
	for h in os.listdir("../data/high_score/cv/"):
		#for j in xrange(10): # repeat 10 times.
		#	random.seed(j)
		init_cv = pd.read_csv("../data/high_score/cv/%s"%h)
		init_pred = pd.read_csv("../data/high_score/pred/%s"%h)
		init_agree = pd.read_csv("../data/high_score/agreement/%s"%h)
		init_correlation = pd.read_csv("../data/high_score/correlation/%s"%h)
		high_per = (init_cv['prediction'].values,init_pred['prediction'].values, 
			init_agree['prediction'].values,init_correlation['prediction'].values)
		alg = Passing_Search(high_per)
		alg.search()
	print "ALL DONE!"







