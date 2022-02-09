import pickle
import numpy as np
# import pandas as pd

# load_data = open(r'../data_after_process_weekday.pickle','rb')
# data_weekday = pickle.load(load_data)
# dataaa = data_weekday

class stand_normal_reverse:
    
    def __init__(self,data=[0]):
        self.data = data
        
    def normal(self):
        "normalization"
        max_v = max(self.data)
        min_v = min(self.data)
        data = (self.data - min_v)/(max_v - min_v)
        return data
     
    def stand(self):
        "Z-Score standardization"
        ave = np.mean(self.data)
        var = np.var(self.data)
        data = (self.data - ave)/var
        return data


def merge_dict(dict_data,n):
    data = np.empty(shape=(len(dict_data[0]),1))
    for i in range(len(dict_data)):
        or_data = np.array(dict_data[i])[:,n].reshape(len(data),1)
        data = np.hstack((data,or_data))            
    return np.transpose(data[:,1:])
            

# c=merge_dict(data_weekday,1)
# n=1
# dict_data = data_weekday
# data = np.empty(shape=(len(dict_data[0]),1))
# # for i in range(len(dict_data)):
# for i in [0]:
#         or_data = np.array(dict_data[i][:,n]).reshape(len(data),1)
#         data = np.hstack((data,or_data)) 