import data_preprocessor as dp
import data_train as dtrain
import pandas as pd



scaler,x_train, x_test, y_train,y_test = dp.get_data('heart.csv')
svm, lr, sklgbm, catboost = dtrain.train_model(x_train,y_train)


def cli_predict():
    print('please submit the following details:')
    pth = input('Please enter the path of your csv file:')
    data = pd.read_csv(f"{pth}")
    data = data.iloc[11:16,1:]
    print('0-means you are safe')
    print('1-means consult the doctor/ take medication')
    print(catboost.predict(data))
    #print(sklgbm.predict(data))
    
cli_predict()    