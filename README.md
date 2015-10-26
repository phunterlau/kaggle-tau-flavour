# kaggle-tau-flavour
Kaggle Flavour of physics solution code for the discussion in [this blog](https://no2147483647.wordpress.com/2015/10/26/yet-another-not-winning-solution-kaggle-flavours-of-physics-for-finding-%CF%84-%E2%86%92-3%CE%BC/)

It was not a top winning solution, the primary reason was I tried my best for `physical sounds` features and avoiding invariant mass feature, and surely, if one overfit on simulated invariant mass, it would have much higher score. The reason why I didn't use invariant mass was discussed in the blog from the link above.

The code is a little bit messy. It has 6 parts:

* **feature engineering** is at `model_selection/feat.py` which contains all important kinematic, pair-wise and quality selection features as mentioned in the blog post.
* **feature selection** is at `feat_selection/` and one can run `append_feats.py` to generate the full long list of features, and let random forest `rf_feat_selection.py` to select them and print out the importance list. The final selected features are used in the feature engineering part.
* **model selection** is a generalized model selection framework at `model_selection` where `search_v1.py` searches in the model directories and optimizes the final submission of xgboost model by the CV score. To run it, corresponding subdirectories should be created. It is created by littleboat and he has the full credit.
* **ensemble** is at `ensemble` which is a simple weighted average of adding NN results (bad KS score) to a good result (UGrad mix with xgboost after model selection)
* **xgboost** parameter grid search is at `xgboost`
* **Neural Network** model building is at `nn/stacking_nn.py` which is a neural network model training on stacking old features. It may take very long time to run. It is used in the final ensemble of boosting the UGrad+xgboost score to +0.0005 AUC.