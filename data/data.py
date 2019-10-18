#%%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def process_data(train,test,lags):
    attr = 'volumn'
    df1 = pd.read_csv(train, encoding='utf-8').fillna(0)#将NA值替换成0
    df2 = pd.read_csv(test, encoding='utf-8').fillna(0)#将NA值替换成0

    #scaler = StandardScaler().fit(df1[attr].values.reshape(-1, 1)) #数据点中每个特征的数值范围可能变化很大，因此，有时将特征的数值范围缩放到合理的大小是非常重要的
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    #lags一组提取元素（一组一组训练）
    train, test = [], []
    for i in range(lags, len(flow1)):
        train.append(flow1[i - lags: i + 1])
    for i in range(lags, len(flow2)):
        test.append(flow2[i - lags: i + 1])
    
    train = np.array(train)#array有shape
    test = np.array(test)
    np.random.shuffle(train)
    
    X_train = train[:, :-1]#剔除所有list中最后一个打乱的数据 作为训练集
    y_train = train[:, -1]#筛选所有list中最后一个数据 作为标签
    X_test = test[:, :-1]
    y_test = test[:, -1]
    
    return X_train, y_train, X_test, y_test, scaler