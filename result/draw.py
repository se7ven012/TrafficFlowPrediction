#%%
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
def plot_results(y_true, y_preds, names, dataset, lag, batch_size):
    d = '2018/4/4'
    x = pd.date_range(d, periods=288, freq='5min')

    my_dpi=200
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.plot(x, y_true, label='True Data',color='red')
    for name, y_pred in zip(names, y_preds):
        ax.plot(x, y_pred, label=name, lw=0.85)
    

    plt.legend()
    plt.grid(True)
    plt.xlabel('Time of Day')
    plt.ylabel('Traffic Flow')

    date_format = mpl.dates.DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()
    
    plt.savefig('result/'+dataset+'_'+str(lag)+'_'+str(batch_size)+'_'+'result.png',figsize=(800/my_dpi, 400/my_dpi), dpi=my_dpi)
    plt.show()

#%%
