import pandas as pd
import numpy as np
from sklearn.cross_validation import *
from sklearn.grid_search import *
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.pipeline import *
from sklearn.preprocessing import *
import evaluation
import xgboost as xgb
import math
from lasagne import layers, nonlinearities
from lasagne.updates import adagrad, nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.neighbors import *
from scipy.optimize import minimize


from hep_ml.losses import BinFlatnessLossFunction, KnnFlatnessLossFunction
from hep_ml.gradientboosting import UGradientBoostingClassifier


class NN1():
    """workaround class because lasagne has no refit options.
       subsequent calls to fit will not refresh the estimator,
       this is the workaround to fix that."""

    def __init__(self, n_features):
        self.n_features = n_features
        self.nn = None

    def fit(self, X, y):
        if self.nn is not None:
            del self.nn
        self.nn = Pipeline([
            ('scaler', StandardScaler()),
            ('ann', NeuralNet(
                layers=[
                    ('input', layers.InputLayer),
                    ('hidden', layers.DenseLayer),
                    ('dropout1', layers.DropoutLayer),
                    ('hidden2', layers.DenseLayer),
                    ('hidden3', layers.DenseLayer),
                    ('output', layers.DenseLayer),
                    ],
                # layer parameters:
                input_shape=(None, self.n_features),
                hidden_num_units=50,  # number of units in hidden layer
                hidden2_num_units=50,
                dropout1_p=0.10,
                hidden3_num_units=25,


                output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
                output_num_units=2,  # 2 target values

                # optimization method:
                update=nesterov_momentum,
                update_learning_rate=0.01,
                update_momentum=0.9,

                regression=False,  # flag to indicate we're dealing with regression problem
                max_epochs=50,  # TRY 50 and 46 epochs!
                verbose=0,
                eval_size=0.10
                ))])
        self.nn.fit(X, y)

    def predict_proba(self, X):
        return self.nn.predict_proba(X)

class NN2():
    def __init__(self, n_features):
        self.n_features = n_features
        self.nn = None

    def fit(self, X, y):
        if self.nn is not None:
            del self.nn
        self.nn = Pipeline([
            ('scaler', StandardScaler()),
            ('ann', NeuralNet(
                layers=[
                    ('input', layers.InputLayer),
                    ('hidden', layers.DenseLayer),
                    ('hidden2', layers.DenseLayer),
                    ('output', layers.DenseLayer),
                    ],
                # layer parameters:
                input_shape=(None, len(features)),
                hidden_num_units=100,  # number of units in hidden layer
                hidden2_num_units=50,

                output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
                output_num_units=2,  # 2 target values

                # optimization method:
                update=nesterov_momentum,
                update_learning_rate=0.01,
                update_momentum=0.9,

                regression=False,  # flag to indicate we're dealing with regression problem
                max_epochs=7,  # TRY 50 and 46 epochs!
                verbose=0,
                eval_size=0.10
                ))])
        self.nn.fit(X, y)

    def predict_proba(self, X):
        return self.nn.predict_proba(X) 


class NN3():
    def __init__(self, n_features):
        self.n_features = n_features
        self.nn = None

    def fit(self, X, y):
        if self.nn is not None:
            del self.nn
        self.nn = Pipeline([
            ('scaler', StandardScaler()),
            ('ann', NeuralNet(
                layers=[
                    ('input', layers.InputLayer),
                    ('hidden', layers.DenseLayer),
                    ('output', layers.DenseLayer),
                    ],
                # layer parameters:
                input_shape=(None, len(features)),
                hidden_num_units=100,  # number of units in hidden layer

                output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
                output_num_units=2,  # 2 target values

                # optimization method:
                update=adagrad,
                # update_learning_rate=0.01,
                # update_momentum=0.9,

                regression=False,  # flag to indicate we're dealing with regression problem
                max_epochs=3,  # TRY 50 and 46 epochs!
                verbose=0,
                eval_size=0.10
                ))])
        self.nn.fit(X, y)

    def predict_proba(self, X):
        return self.nn.predict_proba(X) 


class NN4():
    def __init__(self, n_features):
        self.n_features = n_features
        self.nn = None

    def fit(self, X, y):
        if self.nn is not None:
            del self.nn
        self.nn = Pipeline([
            ('scaler', StandardScaler()),
            ('ann', NeuralNet(
                layers=[
                    ('input', layers.InputLayer),
                    ('dropout1', layers.DropoutLayer),
                    ('hidden', layers.DenseLayer),
                    ('dropout2', layers.DropoutLayer),
                    ('hidden3', layers.DenseLayer),
                    ('hidden4', layers.DenseLayer),
                    ('output', layers.DenseLayer),
                    ],
                # layer parameters:
                input_shape=(None, len(features)),
                dropout1_p = 0.25,
                hidden_num_units=100,  # number of units in hidden layer
                dropout2_p = 0.10,
                hidden3_num_units=50,
                hidden4_num_units=10,


                output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
                output_num_units=2,  # 2 target values

                # optimization method:
                update=adagrad,
                update_learning_rate=0.007,
                # update_momentum=0.9,

                regression=False,  # flag to indicate we're dealing with regression problem
                max_epochs=164,  # TRY 50 and 46 epochs!
                verbose=0,
                eval_size=0.10
                ))])
        self.nn.fit(X, y)

    def predict_proba(self, X):
        return self.nn.predict_proba(X)     


class XGBoostClassifier():

    def __init__(self, num_boost_round=400, **params):
        self.clf = None
        self.num_boost_round = num_boost_round
        self.params = params
        self.params.update({'objective': 'binary:logistic'})

    def fit(self, X, y, num_boost_round=None):
        num_boost_round = num_boost_round or self.num_boost_round
        dtrain = xgb.DMatrix(X, y)
        self.clf = xgb.train(self.params, dtrain, num_boost_round)

    def predict(self, X):
        Y = self.predict_proba(X)
        y = np.argmax(Y, axis=1)
        return y

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


from feat import add_features

def get_train():
    return pd.read_csv('../input/training.csv')


def scipy_opt(blended, test_y, OOS_blended):
    """uses scip.optimize.minimize function to find ensemble coefficients"""

    predictions = []
    for i in range(len(blended[0])):
        predictions.append(blended[:, i])

    def auc_func(weights):
        final_prediction = 0
        for weight, prediction in zip(weights, predictions):
            final_prediction += weight*prediction

        # final_prediction = map(lambda x: 1 if x > 0.5 else 0, final_prediction)
        # return -1.0 * accuracy_score(test_y, final_prediction)
        return -1.0 * evaluation.roc_auc_truncated(test_y, final_prediction)


    # the algorithms need a starting value, right not we chose 0.5 for all weights
    # its better to choose many random starting points and run minimize a few times
    starting_values = [0.5] * len(predictions)
    # adding constraints  and a different solver as suggested by user 16universe
    # https://kaggle2.blob.core.windows.net/forum-message-attachments/75655/2393/otto%20model%20weights.pdf?sv=2012-02-12&se=2015-05-03T21%3A22%3A17Z&sr=b&sp=r&sig=rkeA7EJC%2BiQ%2FJ%2BcMpcA4lYQLFh6ubNqs2XAkGtFsAv0%3D
    cons = ({'type': 'eq', 'fun': lambda w: 1.0 - sum(w)})
    # our weights are bound between 0 and 1
    bounds = [(0, 1)]*len(predictions)

    res = minimize(auc_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons, options={'disp': True, 'maxiter': 2000}, tol=1e-30)

    print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))

    weights = res['x']
    res = np.empty(OOS_blended.shape[0])
    for i in range(res.shape[0]):
        tmp_sum = 0.0
        for j in range(len(weights)):
            tmp_sum += OOS_blended[i][j] * weights[j]
        res[i] = tmp_sum

    return res



def stacked_models(train, features, test, in_sample=True):
    """
    Build stacked generalization models, set in_sample to False
    to predict on test set.
    """

    if in_sample:

        np.random.seed(1)
        new_indices = np.asarray(train.index.copy())
        np.random.shuffle(new_indices)

        train = train.iloc[new_indices].reset_index(drop=True).copy()

        # not used in CV testing..
        del test

        cutoff = int(new_indices.shape[0] * 0.75)

        X_dev = train[:cutoff].reset_index(drop=True).copy()
        Y_dev = train[:cutoff]['signal'].reset_index(drop=True).copy()

        X_test = train[cutoff:][train[cutoff:]['min_ANNmuon'] > 0.4].reset_index(drop=True).copy()
        Y_test = train[cutoff:][train[cutoff:]['min_ANNmuon'] > 0.4]['signal'].reset_index(drop=True).copy()

    else:
        np.random.seed(1)
        new_indices = np.asarray(train.index.copy())
        np.random.shuffle(new_indices)

        train = train.iloc[new_indices].reset_index(drop=True).copy()


        X_dev = train.reset_index(drop=True).copy()
        Y_dev = train['signal'].reset_index(drop=True).copy()

        X_test = test.reset_index(drop=True).copy()
        Y_test = None

    n_folds = 5

    # put ur parameter tuned CLFs in this list.

    clfs = [
        RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=20, n_jobs=-1),
        RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=20, n_jobs=-1, max_depth=6),
        ExtraTreesClassifier(n_estimators=200, criterion='entropy', random_state=50, n_jobs=-1),
        ExtraTreesClassifier(n_estimators=200, criterion='entropy', random_state=50, n_jobs=-1, max_depth=6),
        Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())]),

        UGradientBoostingClassifier(loss=BinFlatnessLossFunction(['mass'], n_bins=15, uniform_label=0), n_estimators=150, subsample=0.1, max_depth=6, min_samples_leaf=10, learning_rate=0.1, train_features=features, random_state=11),
        UGradientBoostingClassifier(loss=KnnFlatnessLossFunction(['mass'], n_neighbours=30, uniform_label=0), n_estimators=150, subsample=0.1, max_depth=6, min_samples_leaf=10, learning_rate=0.1, train_features=features, random_state=11),

        UGradientBoostingClassifier(loss=BinFlatnessLossFunction(['mass'], n_bins=15, uniform_label=0), n_estimators=100, subsample=0.8, max_depth=6, min_samples_leaf=10, learning_rate=0.1, train_features=features, random_state=11),
        UGradientBoostingClassifier(loss=KnnFlatnessLossFunction(['mass'], n_neighbours=30, uniform_label=0), n_estimators=100, subsample=0.8, max_depth=6, min_samples_leaf=10, learning_rate=0.1, train_features=features, random_state=11),



        XGBoostClassifier(eval_metric='auc', objective='binary:logistic',
                          num_class=2,
                          nthread=4,
                          silent=1,

                          colsample_bytree=0.6,
                          eta=0.005,
                          max_depth=6,
                          min_child_weight=13,
                          seed=1337,
                          subsample=0.7
                          ),
        NN1(len(features)),
        NN2(len(features)),
        NN3(len(features)),
        NN4(len(features))
    ]


    skf = list(StratifiedKFold(Y_dev, n_folds))

    # Number of training data x Number of classifiers
    blend_train = np.zeros((X_dev.shape[0], len(clfs)))
    # Number of testing data x Number of classifiers
    blend_test = np.zeros((X_test.shape[0], len(clfs)))

    print 'X_test.shape = %s' % (str(X_test.shape))
    print 'blend_train.shape = %s' % (str(blend_train.shape))
    print 'blend_test.shape = %s' % (str(blend_test.shape))

    # For each classifier, we train the number of fold times (=len(skf))
    for j, clf in enumerate(clfs):
        print 'Training classifier [%s]' % (j)
        # Number of testing data x Number of folds , we will take the mean of
        # the predictions later
        blend_test_j = np.zeros((X_test.shape[0], len(skf)))
        for i, (train_index, cv_index) in enumerate(skf):
            print 'Fold [%s]' % (i)

            # This is the training and validation set
            X_train = X_dev.iloc[train_index].copy()
            Y_train = Y_dev.iloc[train_index].copy()
            X_cv = X_dev.iloc[cv_index].copy()
            Y_cv = Y_dev.iloc[cv_index].copy()

            # handle the case of hep.ml stuff
            if type(clf) == type(UGradientBoostingClassifier()):
                clf.fit(X_train[features + ['mass']], Y_train.values.astype(np.int32))
            else:
                clf.fit(X_train[features], Y_train.values.astype(np.int32))

            # This output will be the basis for our blended classifier to train against,
            # which is also the output of our classifiers
            blend_train[cv_index, j] = clf.predict_proba(X_cv[features])[:, 1]
            blend_test_j[:, i] = clf.predict_proba(X_test[features])[:, 1]
        # Take the mean of the predictions of the cross validation set
        blend_test[:, j] = blend_test_j.mean(1)

    print 'Y_dev.shape = %s' % (Y_dev.shape)

    # blend with LR...
    bclf = LogisticRegression()
    bclf.fit(blend_train, Y_dev)

    bclf2 = GradientBoostingClassifier(n_estimators=150, learning_rate=0.02, max_depth=4, subsample=0.9, verbose=3, random_state=1337)
    bclf2.fit(blend_train, Y_dev)

    bclf3 = NeuralNet(layers=[
                    ('input', layers.InputLayer),
                    ('hidden', layers.DenseLayer),
                    ('output', layers.DenseLayer)],

                    # layer parameters:
                    input_shape=(None, blend_train.shape[1]),
                    hidden_num_units = blend_train.shape[1],


                    output_nonlinearity=nonlinearities.softmax,  # output layer uses identity function
                    output_num_units=2,  # 2 target values

                    # optimization method:
                    update=nesterov_momentum,
                    update_learning_rate=0.01,
                    update_momentum=0.9,

                    regression=False,  # flag to indicate we're dealing with regression problem
                    max_epochs=53,  # TRY 50 and 46 epochs!
                    verbose=1,
                    eval_size=0.10

                    )

    bclf3.fit(blend_train.astype(np.float32), Y_dev.astype(np.int32))

    bclf4 = AdaBoostClassifier(n_estimators=400, random_state=88)
    bclf4.fit(blend_train, Y_dev)

    # Predict now
    Y_test_predict = bclf.predict_proba(blend_test)[:, 1]
    Y_test_predict2 = bclf2.predict_proba(blend_test)[:, 1]
    Y_test_predict3 = bclf3.predict_proba(blend_test.astype(np.float32))[:, 1]
    Y_test_predict4 = bclf4.predict_proba(blend_test)[:, 1]

    print 'Logit Coefs:', bclf.coef_
    if in_sample:
        score = evaluation.roc_auc_truncated(Y_test, Y_test_predict)
        score2 = evaluation.roc_auc_truncated(Y_test, Y_test_predict2)
        score3 = evaluation.roc_auc_truncated(Y_test, blend_test.mean(1))
        score4 = evaluation.roc_auc_truncated(Y_test, scipy_opt(blend_train, Y_dev, blend_test))
        score5 = evaluation.roc_auc_truncated(Y_test, (Y_test_predict + Y_test_predict2) / 2.0)
        score6 = evaluation.roc_auc_truncated(Y_test, Y_test_predict3)
        score7 = evaluation.roc_auc_truncated(Y_test, (Y_test_predict + Y_test_predict2 + Y_test_predict3) / 3.0)
        score8 = evaluation.roc_auc_truncated(Y_test, Y_test_predict4)
        score9 = evaluation.roc_auc_truncated(Y_test, (Y_test_predict2 + Y_test_predict3 + Y_test_predict4) / 3.0)
        score10 = evaluation.roc_auc_truncated(Y_test, (Y_test_predict + Y_test_predict2 + Y_test_predict3 + Y_test_predict4) / 4.0)

        print 'LR Score = %s' % (score)
        print 'GB Score = %s' % (score2)
        print 'MEAN Score = %s' % (score3)
        print 'Scipy Score = %s' % (score4)
        print 'LR + GB score = %s' % (score5)
        print 'ANN Score= %s' % (score6)
        print 'LR + GB + ANN Score = %s' % (score7)
        print 'ADA Score = %s' % (score8)
        print 'GB + ANN + ADA Score = %s' % (score9)
        print 'LR + GB + ANN + ADA Score = %s' % (score10)
        return blend_train, Y_dev, blend_test, Y_test

    # average of ADA, ANN and GBM.
    return (Y_test_predict + Y_test_predict2 + Y_test_predict3 + Y_test_predict4) / 4.0

train = pd.read_csv('../input/training.csv')
test  = pd.read_csv('../input/test.csv')

train = add_features(train)
test = add_features(test)

# add SPDHITS back...

filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal', 'p0_eta','p1_eta','p2_eta','LifeTime',
              'FlightDistanceError']

#features = list(train.columns)
features = list(f for f in train.columns if f not in filter_out)

is_test = False

res = stacked_models(train, features, test, is_test)

if not is_test:

    test['prediction'] = res

    test[['id', 'prediction']].to_csv('layer2_test_V3.csv', index=False)

else:
    tmp = pd.DataFrame()

    train = res[0]
    train_y = res[1]

    test = res[2]
    test_y = res[3]

    for i in range(train.shape[1]):
        tmp['feat_'+str(i)] = train[:, i]

    tmp['train_y'] = train_y
    tmp.to_csv('train2.csv', index=False)

    tmp = pd.DataFrame()

    for i in range(test.shape[1]):
        tmp['feat_'+str(i)] = test[:, i]
    tmp['test_y'] = test_y

    tmp.to_csv('test2.csv', index=False)
