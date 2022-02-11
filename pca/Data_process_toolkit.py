import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    "change dictionary to matrix"
    data = np.empty(shape=(len(dict_data[0]),1))
    for i in range(len(dict_data)):
        or_data = np.array(dict_data[i])[:,n].reshape(len(data),1)
        data = np.hstack((data,or_data))            
    return np.transpose(data[:,1:])

def merge_dict_ft(dict_data,n):
    "change dictionary to matrix"
    data = np.empty(shape=(len(dict_data[0]),0))
    for i in range(len(dict_data)):
        or_data = np.array(dict_data[i]).reshape(len(data),1)
        data = np.hstack((data,or_data))            
    return np.transpose(data[:,1:])
            
def count_ave_in_dict(matrix,class_vector,number_clusters):
    "generate matrix"
    """
    Matrix: original data 
    class_vector: km_selection.labels_
    number_cluster: int, number of clusters
    
    Used in Feature_Selection

    """
    out_dicit = {}
    out_matrix = np.zeros((number_clusters,np.shape(matrix)[1]))
    for i in range(number_clusters):   
        empty = np.empty(shape=(0,np.shape(matrix)[1]))
        for ii in range(len(class_vector)):
            if class_vector[ii] == i:
                out_matrix[i] = matrix[ii,:] + out_matrix[i]
                empty = np.vstack((empty,matrix[ii,:]))
        out_dicit[i] = empty  
    "divide number of smaters in each cluster"
    k_count = pd.value_counts(class_vector)     
    for i in range(len(k_count)):
        for ii in range(len(out_matrix)):
            if i == ii:
                out_matrix[ii,:] = out_matrix[ii,:]/k_count[i]
    return out_matrix,out_dicit

def draw_graph_km(matrix_ave,dict_cluster):
    """
    matrix_ave: AVE value 
    dict_cluster: 
    
    Used in Feature_Selection, G
    """
    for i in range(len(matrix_ave)):
        for ii in range(len(dict_cluster[i])):
            plt.plot(dict_cluster[i][ii,:],  alpha=.1, color='black')
        title = 'Consumption profile of '+str(i)+ ' '+str(max(dict_cluster))+'clusters'
        plt.title(title)
        plt.plot(matrix_ave[i], color='red')
        plt.show()

def trace_original_data_g(matrix_data,matrix_labels,n):
    dicit_ordered_g = {}
    for i in range(n):
        empty = np.empty((0,np.shape(matrix_data)[1]))
        for ii in range(len(matrix_labels)):
            if matrix_labels[ii] == i:
                empty = np.vstack((empty,matrix_data))
            dicit_ordered_g[i] = empty
    return dicit_ordered_g

