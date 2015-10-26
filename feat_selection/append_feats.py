#put all features to each file, and give random forest for feature selection
import sys
import pandas as pd
from feat_all import add_features

train_eval = pd.read_csv("training_eval_only.csv")
train_eval = add_features(train_eval)
train_eval.to_csv('feat_training_eval_only.csv',index=False)
print 'finish train_eval'

train = pd.read_csv("training.csv")
train = add_features(train)
train.to_csv('feat_training.csv',index=False)
print 'finish training'

test = pd.read_csv("test.csv")
test = add_features(test)
test.to_csv('feat_test.csv',index=False)
print 'finish test'

check_agreement = pd.read_csv('check_agreement.csv')
check_agreement = add_features(check_agreement)
check_agreement.to_csv('feat_check_agreement.csv',index=False)
check_correlation = pd.read_csv('check_correlation.csv')
check_correlation = add_features(check_correlation)
check_correlation.to_csv('feat_check_correlation.csv',index=False)
