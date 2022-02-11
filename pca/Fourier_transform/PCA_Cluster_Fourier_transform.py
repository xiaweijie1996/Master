import sys
sys.path.append(r'../')
from Data_process_toolkit import *
from Fourier_transform import *
sys.path.append(r'../Featrue_Selection')
from Feature_Selection import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import pandas as pd
import copy

load_data = open(r'../../data_after_process_weekday.pickle','rb')
data_weekday = pickle.load(load_data)
or_data_v,or_data_h = FFT_dicit(data_weekday,1,0)

"Process data, add a index to each smart meter, normalization of the data"
data_ft = copy.deepcopy(merge_dict_ft(or_data_h,0))
or_data_nochange = merge_dict(data_weekday,1)
for i in range(np.shape(or_data_nochange)[0]):
    or_data_nochange[i,:] = stand_normal_reverse(or_data_nochange[i,:]).normal()
# data_ft = np.vstack((back_up_data_ft,back_up_data_ft))

# "Conduct PCA for data from Ft, find the knee point of PCA, k=97 "
# pca_ft = PCA(n_components = 192) #'mle'
# pca_ft.fit(data_ft)
# pca_ft_var_sum = sum(pca_ft.explained_variance_)
# pca_ratio_var_ft = []
# for n in range(192):
#     pca_ft_n = PCA(n_components=n) #'mle'
#     pca_ft_n.fit(data_ft)
#     pca_ft_var_n = sum(pca_ft_n.explained_variance_)
#     print(n)
#     pca_ratio_var_ft.append(pca_ft_var_n*100/pca_ft_var_sum)

# y_var = pca_ratio_var_ft
# x_var = range(1,len(y_var)+1)
# plt.title('Relationship between components and percentage of information FS')
# plt.xlabel('number of components')
# plt.ylabel('perentage of information')
# plt.plot(x_var,y_var)
# plt.show()

# "find the knee potion l=97, generate the data after PCA"
# pca_ft = PCA(n_components=97) #'mle'
# pca_ft.fit(data_ft)
# data_pca_ft = pca_ft.transform(data_ft)

# "conduct Kmeans for ft data, find the knee point n=7"
# km_inertia_record_ft = []
# for n in range(1,98):
#     km_ft = KMeans(n_clusters=n,random_state=0).fit(data_pca_ft)
#     km_centers_ft = km_ft.cluster_centers_
#     km_labels_ft = km_ft.labels_
#     km_inertia_ft = km_ft.inertia_
#     km_inertia_record_ft.append(km_inertia_ft)
#     print(n)
#     # km_n_iter_Selection = km_selection.n_iter_
    
# y_km_ft = km_inertia_record_ft
# x_km_ft = range(1,1+len(y_km_ft))
# plt.title('Relationship between K and inertia')
# plt.xlabel('K number of clusters')
# plt.ylabel('Loss inertia')
# plt.plot(x_km_ft,y_km_ft)
# plt.show()

"cluster 99 smart meters into 5-6 group, for each group, draw the graph to exlpore characteristices"
n_clusters = 3
km_ft = KMeans(n_clusters,random_state=0).fit(data_ft)
km_centers_ft = km_ft.cluster_centers_
km_labels_ft = km_ft.labels_

"Study the percentage of each cluster"
km_count_ft = pd.value_counts(km_labels_ft)
km_out_matrix_ft,km_dicit_ft = count_ave_in_dict(data_ft,km_labels_ft,n_clusters)
draw_graph_km(km_out_matrix_ft,km_dicit_ft)

"Darw grap of original data order"
# dicit_ordered_g = trace_original_data_g(back_up_data_g,km_labels_g,n_clusters)
km_out_matrix_ft,km_out_dicit_ft = count_ave_in_dict(or_data_nochange,km_labels_ft,n_clusters)
draw_graph_km(km_out_matrix_ft,km_out_dicit_ft)    