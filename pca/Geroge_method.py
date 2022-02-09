import pickle
import numpy as np
import pandas as pd

load_data = open(r'../data_after_process_weekday.pickle','rb')
data_weekday = pickle.load(load_data)

def rearrange_order_data(data):
    "add index to every time slot"
    for x in range(len(data)):
        array_1 = np.arange(len(data[x])).reshape((len(data[x]),1))
        data[x] = np.hstack((array_1,data[x]))
    
    "list from big to small by 2"
    for x in range(len(data)):
        df = pd.DataFrame(data[x])
        df = df.sort_values(by=2,ascending=False)
        data[x] = df
    return data
        
