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



def train_model(x_train,y_train):

    svm = SVC( random_state=0)
    svm = svm.fit(x_train,y_train)

    lr = LogisticRegression(solver='liblinear', C=4.2, max_iter=5, penalty='l2', random_state=0)
    lr = lr.fit(x_train, y_train)

    sklgbm = GradientBoostingClassifier(n_estimators=250, learning_rate=0.15, max_depth=6, max_features='sqrt', random_state=0)
    sklgbm = sklgbm.fit(x_train, y_train)

    catboost = CatBoostClassifier(iterations=200, depth=9, learning_rate=0.2,bootstrap_type='Bernoulli', random_state=0)
    catboost = catboost.fit(x_train, y_train)
    

    return svm, lr, sklgbm, catboost