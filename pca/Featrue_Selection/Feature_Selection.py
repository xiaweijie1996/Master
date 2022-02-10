import numpy as np
import pandas as pd
# import pickle
# load_data = open(r'../data_after_process_weekday.pickle','rb')
# data_weekday = pickle.load(load_data)

def feature_selection(data):
    """
    Input: a dictionary, the key is the number of smart meter
    Output: a dictionary,but the selected data in the end of each table
    """
    
    "add index to every time slot"
    for x in range(len(data)):
        array_1 = np.arange(len(data[x])).reshape((len(data[x]),1))
        data[x] = np.hstack((array_1,data[x]))
        
    for x in range(len(data)):
       
        df = pd.DataFrame(data[x])
        dff = df.sort_values(by=2,ascending=False)
        
        # "find minmum vaule and position"
        # max_v = dff.iloc[0,2]
        # max_index = dff.iloc[0,0]/1000
   
        # "find maxmum vaule and position"
        # min_v = dff.iloc[-1,2]
        # min_index = dff.iloc[-1,0]/1000
        
        # "combine data"
        # n = len(data[x])
        # df.loc[n] = [n,0,max_v]
        # df.loc[n+1] = [n+1,0,max_index ]
        # df.loc[n+2] = [n+2,0,min_v]
        # df.loc[n+3] = [n+3,0,min_index]
        data[x] = df
        
    return data 

# a= feature_selection(data_weekday)
