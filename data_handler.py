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
    

