import sys
sys.path.append(r'../')
from Normalization_Standardization import *
from Feature_Selection import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle

load_data = open(r'../../data_after_process_weekday.pickle','rb')
data_weekday = pickle.load(load_data)
or_data = feature_selection(data_weekday)

"perfor normalization"

data_selection = merge_dict(or_data,2)
# data_selection = np.vstack((data_selection,data_selection))

"PCA for Feature_Selection "
pca_selection = PCA(n_components=98) #'mle'
pca_selection.fit(data_selection)
pca_selec_var_sum = sum(pca_selection.explained_variance_)
pca_ratio_var_selection = []
for n in range(99):
    pca_selection_n = PCA(n_components=n) #'mle'
    pca_selection_n.fit(data_selection)
    # data_pca_selection = pca_selection.transform(data_selection)
    # pca_selec_ratio = pca_selection.explained_variance_ratio_
    pca_selec_var_n = sum(pca_selection_n.explained_variance_)
    print(n)
    pca_ratio_var_selection.append(pca_selec_var_n*100/pca_selec_var_sum)

y_var = pca_ratio_var_selection
x_var = range(1,len(y_var)+1)
plt.title('Relationship between components and percentage of information')
plt.xlabel('number of components')
plt.ylabel('perentage of information')
plt.plot(x_var,y_var)
plt.show()

"the final data"
pca_selection = PCA(n_components=90) #'mle'
pca_selection.fit(data_selection)
data_pca_selection = pca_selection.transform(data_selection)

"Kmeans for SELECTION data"
km_inertia_record_selection = []
for n in range(1,99):
    km_selection = KMeans(n_clusters=n,random_state=0).fit(data_selection)
    km_centers_selection = km_selection.cluster_centers_
    km_labels_selection = km_selection.labels_
    km_inertia_selection = km_selection.inertia_
    km_inertia_record_selection.append(km_inertia_selection)
    print(n)
    # km_n_iter_Selection = km_selection.n_iter_
    
y_km_selection = km_inertia_record_selection
x_km_selection = range(1,1+len(y_km_selection))
plt.title('Relationship between K and inertia')
plt.xlabel('K number of clusters')
plt.ylabel('Loss inertia')
plt.plot(x_km_selection,y_km_selection)
plt.show()