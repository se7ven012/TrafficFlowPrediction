#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_accuracy():
    my_dpi = 200

    names = ['LSTM','GRU','RNN']
    col=[2,5,8]

    for name, col in zip(names,col):
        MAE=pd.read_csv('accuracy.csv',usecols=[col])
        print(MAE)

        fig = plt.figure()
        plt.xlabel('Depth')
        plt.ylabel('MAE')

        ax = fig.add_subplot(111)
        epoch=np.arange(1,7,1)
        ax.plot(epoch,MAE,label=name)
        plt.legend()
        plt.savefig(name+'_'+'accuracy.png',figsize=(800/my_dpi, 400/my_dpi), dpi=my_dpi)
        plt.show()
        print(name+'+'+'finished')
    print('Done!')

plot_accuracy()
#%%
