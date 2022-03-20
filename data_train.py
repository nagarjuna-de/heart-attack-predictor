## Import libraries
import data_preprocessor as dp

from sklearn import set_config
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline  
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler


from sklearn.ensemble      import GradientBoostingClassifier
from catboost              import CatBoostClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.svm           import SVC



scaler,x_train, x_test, y_train,y_test = dp.get_data('heart.csv')

def train_model(x_train,y_train):

    svm = SVC(kernel="linear", random_state=0)
    svm = svm.fit(x_train,y_train)

    lr = LogisticRegression(random_state=0)
    lr = lr.fit(x_train, y_train)

    sklgbm = GradientBoostingClassifier(n_estimators=100, random_state=0)
    sklgbm = sklgbm.fit(x_train, y_train)

    catboost = CatBoostClassifier(n_estimators=100, random_state=0)
    catboost = catboost.fit(x_train, y_train)
    

    return svm, lr, sklgbm, catboost