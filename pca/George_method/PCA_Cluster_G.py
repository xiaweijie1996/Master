import sys
sys.path.append(r'../')
from Data_process_toolkit import *
from Geroge_method import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle
import pandas as pd
import copy

load_data = open(r'../../data_after_process_weekday.pickle','rb')
data_weekday = pickle.load(load_data)
ordered_data, or_data= rearrange_order_data(data_weekday)

"Process data, add a index to each smart meter, normalization of the data"
back_up_data_g = merge_dict(or_data,2)
data_g = merge_dict(ordered_data,2)
for i in range(np.shape(data_g)[0]):
    data_g[i,:] = stand_normal_reverse(data_g[i,:]).normal()
    back_up_data_g[i,:] = stand_normal_reverse(back_up_data_g[i,:]).normal()
for i in range(len(data_g)):
    plt.plot(data_g[i])
plt.title('Rearrange the order of the data from big to small')
plt.show()

"Conduct PCA for data from G, find the knee point of PCA "
# pca_g = PCA(n_components=96) #'mle'
# pca_g.fit(data_g)
# pca_var_sum_g = sum(pca_g.explained_variance_)
# pca_ratio_var_g = []
# for n in range(96):
#     pca_g_n = PCA(n_components=n) #'mle'
#     pca_g_n.fit(data_g)
#     # data_pca_selection = pca_selection.transform(data_selection)
#     # pca_selec_ratio = pca_selection.explained_variance_ratio_
#     pca_var_n_g = sum(pca_g_n.explained_variance_)
#     print(n)
#     pca_ratio_var_g.append(pca_var_n_g*100/pca_var_sum_g)
    
# y_var = pca_ratio_var_g
# x_var = range(1,len(y_var)+1)
# plt.title('Relationship between components and percentage of information g')
# plt.xlabel('number of components')
# plt.ylabel('perentage of information')
# plt.plot(x_var,y_var)
# plt.show()

# "find the knee potion l=90, generate the data after PCA"
# pca_g = PCA(n_components=90) #'mle'
# pca_g.fit(data_g)
# data_pca_g = pca_g.transform(data_g)

"conduct Kmeans for G data, find the knee point n=7"
# km_inertia_record_g = []
# for n in range(1,99):
#     km_g = KMeans(n_clusters=n,random_state=0).fit(data_g)
#     km_centers_g = km_g.cluster_centers_
#     km_labels_g = km_g.labels_
#     km_inertia_g = km_g.inertia_
#     km_inertia_record_g.append(km_inertia_g)
#     print(n)
#     # km_n_iter_Selection = km_selection.n_iter_
    
# y_km_g = km_inertia_record_g
# x_km_g = range(1,1+len(y_km_g))
# plt.title('Relationship between K and inertia G')
# plt.xlabel('K number of clusters')
# plt.ylabel('Loss inertia')
# plt.plot(x_km_g,y_km_g)
# plt.show()

"cluster 99 smart meters into 5-6 group, for each group, draw the graph to exlpore characteristices"
n_clusters = 4
km_g = KMeans(n_clusters,random_state=0).fit(data_g)
km_centers_g = km_g.cluster_centers_
km_labels_g = km_g.labels_

"Study the percentage of each cluster"
km_count_g = pd.value_counts(km_labels_g)
km_out_matrix_ordered_g,km_dicit_ordered_g = count_ave_in_dict(data_g,km_labels_g,n_clusters)
draw_graph_km(km_out_matrix_ordered_g,km_dicit_ordered_g)

"Darw grap of original data order"
# dicit_ordered_g = trace_original_data_g(back_up_data_g,km_labels_g,n_clusters)
km_out_matrix_g,km_out_dicit_g = count_ave_in_dict(back_up_data_g,km_labels_g,n_clusters)
draw_graph_km(km_out_matrix_g,km_out_dicit_g)         