#%%
import numpy as np
import pandas as pd
from pandas import DataFrame
from keras.models import load_model
from data.data import process_data
from result.draw import plot_results
from metrics.metrics import MAPE,eva_regress
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def main():
    batch_sizes=[128,256,512]
    lag = [6,9,12,18]
    names=['LSTM','GRU','RNN']
    datasets=['workdays']
    net = []

    df = DataFrame({
        'batch_size':[0],
        'lag':[0],
        'dataset':[0],
        'network':[0],
        'MAPE':[0],
        'MAE':[0],
        'RMSE':[0]}, columns= 
        ['batch_size','lag', 'dataset','network','MAPE','MAE','RMSE'])

    for batch_size in batch_sizes:
        for itera in lag:
            for dataset in datasets:
                file1 = 'data/'+ dataset + '_' +'train.csv'
                file2 = 'data/'+ dataset + '_' +'test.csv' 

                for name in names:
                    net.append(load_model('model/' + dataset + '_' + name + '_' + str(itera) + '_' + str(batch_size)+'.h5'))

                #obtain data
                X_train, y_train, X_test, y_test, scaler = process_data(file1, file2, itera)
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(1, -1)[0]

                y_preds = []

                for name,model in zip(names,net):
                    predicted = model.predict(X_test)
                    predicted = scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
                    y_preds.append(predicted[:288])
                    result = eva_regress(y_test, predicted, name ,dataset ,itera, batch_size)
                    df = pd.concat([df,result],ignore_index=True)

                plot_results(y_test[: 288],y_preds,names,dataset,itera,batch_size)
            net=[]
    print('Done!')
    df.to_csv (r'metrics_result.csv', index = None, header=True)

main()

#%%
