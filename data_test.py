import data_preprocessor as dp
import data_train as dtrain

from sklearn import set_config
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline  
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


from sklearn.ensemble      import GradientBoostingClassifier
from catboost              import CatBoostClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import SVC

scaler,x_train, x_test, y_train,y_test = dp.get_data('heart.csv')
svm, lr, sklgbm, catboost = dtrain.train_model(x_train,y_train)

def test_model (x_test, y_test):
    svm_pred = svm.predict(x_test)
    lr_pred = lr.predict(x_test)
    sklgbm_pred = sklgbm.predict(x_test)
    catboost_pred = catboost.predict(x_test)

    svm_score = accuracy_score(y_test, svm_pred)
    lr_score = accuracy_score(y_test, lr_pred)
    sklgbm_score = accuracy_score(y_test, sklgbm_pred)
    catboost_score = accuracy_score(y_test, catboost_pred)

    return svm_score, lr_score, sklgbm_score, catboost_score

svm_score, lr_score, sklgbm_score, catboost_score = test_model(x_test, y_test)   

print(f"svm:{svm_score}\n lr:{lr_score}\n sklgbm:{sklgbm_score}\n catboost:{catboost_score}")
