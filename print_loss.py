#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame

def plot_results():
    my_dpi = 200
    lags = [6,9,12,18]
    batch_sizes = [128,256,512]
    dataset = ['workdays']
    names = ['LSTM','GRU','RNN']

    plt.figure()
    
    for name in names:
        i=0
        for lag in lags:
            plt.subplot(221+i)
            plt.tight_layout()
            scale_ls=[0.0020,0.0035,0.005,0.010,0.015]

            plt.title('Time steps ='+' '+str(lag),fontsize=8)
            plt.xticks(fontsize=7)
            plt.yticks(scale_ls,fontsize=7)
            plt.xlabel('Epoch',fontsize=7)
            plt.ylabel('Loss',fontsize=7)

            i+=1
            for batch_size in batch_sizes:
                Loss=pd.read_csv('model/'+dataset[0] + '_' + name + '_' + str(lag) + '_' + str(batch_size)+ ' loss.csv',usecols=[1])
                epoch=np.arange(0,300,1)
                plt.plot(epoch,Loss,label='batch size='+str(batch_size),lw=0.6)
                plt.legend(fontsize=8)
        plt.savefig('result/'+dataset[0]+'_'+name+'_'+'loss.png',figsize=(800/my_dpi, 400/my_dpi), dpi=my_dpi)
        plt.show()
    print('Finished!')

plot_results()
#%%
