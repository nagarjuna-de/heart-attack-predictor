## Import librarires required for data preprocessing.
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from numpy.testing._private.utils import decorate_methods

from sklearn.pipeline import Pipeline      # Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

## making the function for data preprocessing
def get_data(pth):
    data = pd.read_csv(pth)
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace = True)

    ## chainging numerical col 'chol' into categorical column:
    data['chol']= pd.cut(data['chol'], bins=[0,200,239,564], labels = [0,1,2]) #0-High, 1-Borderline high, 2-High
    data['trtbps']= pd.cut(data['trtbps'], bins=[90,120,139,200], labels = [0,1,2])
    data['age'] = pd.cut(data['age'], bins=[25,53,80], labels = [0, 1])
    data = data.astype({'trtbps':'float64','chol':'float64', 'age':'int'})

    ### Creating new column:
    data['chol_bps'] = data['chol']+data['trtbps']
    data.drop(['chol','trtbps'], axis=1, inplace=True)

    new_df = data.copy()
    def get_data():
        gen_data = new_df
        for restecg_values in new_df['restecg'].unique():
            new_data = gen_data[gen_data['restecg']== restecg_values]
            thalachh_std = new_data['thalachh'].std()
            oldpeak_std = new_data['oldpeak'].std()

            for i in gen_data[gen_data['restecg']== restecg_values].index:
                if np.random.randint(2)==1:
                    gen_data['thalachh'].values[i] += thalachh_std/10   
                else:
                    gen_data['thalachh'].values[i] -= thalachh_std/10
                if np.random.randint(2)==1:
                    gen_data['oldpeak'].values[i] += oldpeak_std/10
                else:
                    gen_data['oldpeak'].values[i] -= oldpeak_std/10
        return gen_data 
    std_data = get_data()

    ### Splitting the data
    x= data.drop(['output'], axis=1)
    y = data.output


    x_train, x_test, y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)

    ## Data agumentation to original data

    extra_sample = std_data.sample(std_data.shape[0]//3)
    x_train = pd.concat([x_train, extra_sample.drop(['output'], axis=1)])
    y_train = pd.concat([y_train, extra_sample['output']])

    cat_fe = [ 'sex','cp','fbs','restecg','exng','slp','caa','thall','age']
    num_fe = ['thalachh','oldpeak','chol_bps']

    # Scaling numerical columns
    s_col = ['oldpeak', 'chol_bps','thalachh']
    scaler = MinMaxScaler()
    for d in s_col:
        train_array = x_train[d].to_numpy()
        test_array = x_test[d].to_numpy()
        train_array = train_array.reshape(-1,1)
        test_array = test_array.reshape(-1,1)

        scaler = scaler.fit(train_array)
        x_train[d] = scaler.transform(train_array)
        x_test[d] = scaler.transform(test_array)

    return scaler, x_train, x_test, y_train,y_test


