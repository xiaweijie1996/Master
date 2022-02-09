import pickle
import numpy as np
import pandas as pd

load_data = open(r'../data_after_process_weekday.pickle','rb')
data_weekday = pickle.load(load_data)


def feature_selection(data):
    "add index to every time slot"
    for x in range(len(data)):
        array_1 = np.arange(len(data[x])).reshape((len(data[x]),1))
        data[x] = np.hstack((array_1,data[x]))
        
    for x in range(len(data)):
       
        df = pd.DataFrame(data[x])
        dff = df.sort_values(by=1,ascending=False)
        
        "find minmum vaule and position"
        max_v = dff.loc[0,2]
        max_index = dff.loc[0,0]
   
        "find maxmum vaule and position"
        min_v = dff.iloc[-1,2]
        min_index = dff.iloc[-1,0]
        
        "combine data"
        n = len(data[x])
        df[n+1] = [n+1,0,max_v]
        df[n+2] = [n+1,0,max_index ]
        df[n+3] = [n+1,0,min_v]
        df[n+4] = [n+1,0,min_index]
        
    return data 

# a= feature_selection(data_weekday)
x=0
data=data_weekday[x]
df = pd.DataFrame(data)
dff = df.sort_values(by=1,ascending=False)

"find minmum vaule and position"
max_v = dff.loc[0,1]
max_index = dff.loc[0,0]
   
"find maxmum vaule and position"
min_v = dff.iloc[-1,1]
min_index = dff.iloc[-1,0]

"combine data"
n = len(data)
df[n+1] = [n+1,max_v]
# df[n+2] = [n+1,0,max_index ]
# df[n+3] = [n+1,0,min_v]
# df[n+4] = [n+1,0,min_index]