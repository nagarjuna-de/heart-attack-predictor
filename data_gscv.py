import data_preprocessor as dp
import numpy as np

from sklearn.ensemble      import GradientBoostingClassifier
from catboost              import CatBoostClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

scaler,x_train, x_test, y_train,y_test = dp.get_data('heart.csv')

## call the required function to perform Grid search cv of that particular model.

## GridsearchCV for CatBoostClassifier
def cbc_searchcv():
    cbc = CatBoostClassifier()
    param = {'depth' : [4,5,6,7,8,9, 10],
            'learning_rate' : [0.01,0.02,0.03,0.04, 0.1, 0.2],
            'bootstrap_type' : ['Bayesian','Bernoulli','MVS'],
            ''
            'iterations'    : [10, 20,30,40,50,60,70,80,90, 100,150,200]}

    grid_cbc = GridSearchCV(estimator=cbc, param_grid = param, cv = 2, n_jobs=-1)
    grid_cbc.fit(x_train,y_train)


    print(" Results from Grid Search " )
    print("\n The best estimator across ALL searched params:\n",grid_cbc.best_estimator_)
    print("\n The best score across ALL searched params:\n",grid_cbc.best_score_)
    print("\n The best parameters across ALL searched params:\n",grid_cbc.best_params_)

## GridsearchCV for SVC
def svm_searchcv():
    svm= SVC()
    svc_params = {"decision_function_shape": ["ovo", "ovr"],
                "cache_size": [1, 2, 3, 4, 5, 10,40, 50, 100, 200, 500],
                "gamma": [0.001,0.01,0.1,1,10],
                'C':[0.1,1,10,50,100,150],
                "kernel": ["linear", "poly", "rbf", "sigmoid"]}
    grid_svm = GridSearchCV(estimator=svm, param_grid=svc_params, cv=2, n_jobs=-1)
    grid_svm.fit(x_train,y_train)
    print(" Results from Grid Search " )
    print("\n The best estimator across ALL searched params:\n",grid_svm.best_estimator_)
    print("\n The best score across ALL searched params:\n",grid_svm.best_score_)
    print("\n The best parameters across ALL searched params:\n",grid_svm.best_params_)

## GridsearchCV for LogisticRegression:
def lr_searchcv():
    lr = LogisticRegression()

    param_lr = {'penalty' : ['l1', 'l2','elasticnet'],
                'C' : np.logspace(-2, 2, 20),
                'max_iter':[1,5,10,15,50,100],
                'solver' : ['newton-cg','lbfgs','liblinear','saga']}

    grid_lr = GridSearchCV(estimator=lr, param_grid=param_lr, cv=2, n_jobs=-1)
    grid_lr.fit(x_train,y_train)

    print(" Results from Grid Search " )
    print("\n The best estimator across ALL searched params:\n",grid_lr.best_estimator_)
    print("\n The best score across ALL searched params:\n",grid_lr.best_score_)
    print("\n The best parameters across ALL searched params:\n",grid_lr.best_params_)

## GridsearchCV for Gradient Boosting classifier.
def gbm_searchcv():
    gbm = GradientBoostingClassifier()

    param_gbm = {'learning_rate':[2,1,0.15,0.1,0.05], 
                'n_estimators':[10, 50, 100,250],
                'max_depth':[3,4,5,6,7,8,9],
                'max_features':['auto','sqrt','log2']}
                
    grid_gbm = GridSearchCV(estimator=gbm, param_grid=param_gbm, cv=2, n_jobs=-1)
    grid_gbm.fit(x_train,y_train)

    print(" Results from Grid Search " )
    print("\n The best estimator across ALL searched params:\n",grid_gbm.best_estimator_)
    print("\n The best score across ALL searched params:\n",grid_gbm.best_score_)
    print("\n The best parameters across ALL searched params:\n",grid_gbm.best_params_)
