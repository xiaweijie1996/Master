# import pickle
import numpy as np
import pandas as pd
import copy

# load_data = open(r'../data_after_process_weekday.pickle','rb')
# data_weekday = pickle.load(load_data)

def rearrange_order_data(data):
    """
    Input: a dictionary, the key is the number of smart meter
    Output: a dictionary, but with index of each time slot for each sample
    The table is sorted from larg value to small value
    
    """
    
    "add index to every time slot"
    for x in range(len(data)):
        array_1 = np.arange(len(data[x])).reshape((len(data[x]),1))
        data[x] = np.hstack((array_1,data[x]))
    data_or = copy.deepcopy(data)
    "list from big to small by 2"
    for x in range(len(data)):
        df = pd.DataFrame(data[x])
        df = df.sort_values(by=2,ascending=False)
        data[x] = df
    return data,data_or
        
