#%%
import numpy as np
import pandas as pd
from data.data import process_data
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.models import Sequential
from keras.utils.vis_utils import plot_model
from keras import losses
import matplotlib.pyplot as plt

def LSTM_model(structure):
    model = Sequential()
    model.add(LSTM(structure[1], input_shape=(structure[0], 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(structure[2], return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(structure[3], activation='sigmoid'))
    model.compile(loss=losses.mean_squared_error, optimizer="nadam") 
    return model

def GRU_model(structure):
    model = Sequential()
    model.add(GRU(structure[1], input_shape=(structure[0], 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(structure[2], return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(structure[3], activation='sigmoid'))
    model.compile(loss=losses.mean_squared_error, optimizer="nadam")
    return model

def RNN_model(structure):
    model = Sequential()
    model.add(SimpleRNN(structure[1], input_shape=(structure[0], 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(structure[2], return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(structure[3], activation='sigmoid')) 
    model.compile(loss=losses.mean_squared_error, optimizer="nadam") 
    return model

def train(model,x_train,y_train,config,name,lag,dataset):
    hist = model.fit(
    x_train, y_train,
    batch_size=config["batch"],
    epochs=config["epochs"],
    validation_split=0.05)

    #save model
    model.save('model/' + dataset + '_' + name + '_' + str(lag) + '_' + str(config["batch"])+'.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('model/' + dataset + '_' + name + '_' + str(lag) + '_' + str(config["batch"])+ ' loss.csv', encoding='utf-8', index=False)

def main():
    lags = [6,9,12,18]
    config = {"batch": 256, "epochs": 300}
    datasets=['workdays']
    names=['LSTM','GRU','RNN']
    
    for lag in lags:
        for dataset in datasets:
            file1 = 'data/'+ dataset + '_' +'train.csv'
            file2 = 'data/'+ dataset + '_' +'test.csv'  
            
            x_train, y_train, x_test, y_test, scaler = process_data(file1, file2, lag)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            #为了训练添加维度 添加第三个维度

            netstructure=[lag,64,64,1]
            for name in names:
                if name=='LSTM':
                    m=LSTM_model(netstructure)
                    train(m,x_train,y_train,config,name,lag,dataset)
                    plot_model(m,to_file='result/LSTM'+'_'+ str(lag) + '.png',show_shapes=True,show_layer_names=False,rankdir='TB')
                if name=='GRU':
                    m=GRU_model(netstructure)
                    train(m,x_train,y_train,config,name,lag,dataset)
                    plot_model(m,to_file='result/GRU'+'_'+ str(lag) + '.png',show_shapes=True,show_layer_names=False,rankdir='TB')
                if name=='RNN':
                    m=RNN_model(netstructure)
                    train(m,x_train,y_train,config,name,lag,dataset)
                    plot_model(m,to_file='result/RNN'+'_'+ str(lag) + '.png',show_shapes=True,show_layer_names=False,rankdir='TB')
            print(str(dataset)+':Finished!')
        print(str(lag)+':Done!!')

main()
    

#%%
