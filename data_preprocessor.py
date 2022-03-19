## Import librarires required for data preprocessing.
import pandas as pd
import numpy as np
from numpy.testing._private.utils import decorate_methods

from sklearn.pipeline import Pipeline      # Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

## making the function for data preprocessing
def get_data(pth):
    data = pd.read_csv(pth)
    data['age_class'] = pd.cut(data['age'], bins=[29,44,59,80], labels = [0, 1, 2])
    data['chol']= pd.cut(data['chol'], bins=[0,200,239,564], labels = [0,1,2])
    data['trtbps']= pd.cut(data['trtbps'], bins=[94,120,129,139,159,200], labels = [0,1,2,3,4])
    ## Data enhancement



    ######

    cat_vars = ['age','sex',	cp	trtbps	chol	fbs	restecg	thalachh	exng	oldpeak	slp	caa]
    num_vars = []
