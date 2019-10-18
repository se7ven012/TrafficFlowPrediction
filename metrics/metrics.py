#%%
import math    
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
from pandas import DataFrame

#%%
def MAPE(y_true, y_pred):
    y = [x for x in y_true if x > 0]
    y_pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]

    num = len(y_pred)
    sums = 0

    for i in range(num):
        tmp = abs(y[i] - y_pred[i]) / y[i]
        sums += tmp

    mape = sums * (100 / num)

    return mape

def eva_regress(y_true, y_pred, modelname, dataset, lag, batch_size):
    mape = MAPE(y_true, y_pred)
    vs = metrics.explained_variance_score(y_true, y_pred)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    result = pd.DataFrame([[batch_size,lag,dataset,modelname,round(mape,2),round(mae,2),round(math.sqrt(mse),2)]],columns=['batch_size','lag','dataset','network','MAPE','MAE','RMSE'])

    print(dataset+'_'+modelname+':')
    print('explained_variance_score:%f' % vs)
    print('mape:%f%%' % mape)
    print('mae:%f' % mae)
    print('mse:%f' % mse)
    print('rmse:%f' % math.sqrt(mse))
    print('r2:%f' % r2)

    return result

