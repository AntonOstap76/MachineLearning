import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from spiral_parser import spiral_data_parser

#s1
# # read file
# s1_dataset = pd.read_fwf('s1.txt', header=None, names=['x', 'y'])
#
# # dropping row with no data
# s1_dataset.apply(pd.to_numeric, errors='coerce').dropna()
#
# # ploting dataset
# x = s1_dataset['x']
# y = s1_dataset['y']
# plt.scatter(x, y)
# plt.show()
#
# # finding wcss values for elbow method in order to find optimal number of clusters
# wcss = []
# K = range(5, 20)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k, init="k-means++", n_init=100, max_iter=13, random_state=2)
#     kmeanModel.fit(s1_dataset)
#     wcss.append(kmeanModel.inertia_)
#
# # ploting elbow
# plt.plot(K, wcss, 'bx-')
# plt.xlabel('number of clusters')
# plt.ylabel('Distortion')
# plt.title('Elbow method for s1')
# plt.show()
#
# optimal=15
# hsv_modified = cm.get_cmap('rainbow', optimal)
# kmeans = KMeans(n_clusters=optimal, init="k-means++", n_init=100, max_iter=10, random_state=2)
# kmeans.fit(s1_dataset)
# cluster_predict = kmeans.fit_predict(s1_dataset)
# cluster_centers = kmeans.cluster_centers_
# x2 = [x[0] for x in cluster_centers]
# y2 = [x[1] for x in cluster_centers]
#
# plt.scatter(x, y, cmap=hsv_modified, c=cluster_predict)
# plt.scatter(x2, y2, marker='*', s=500, c='k')
# plt.show()


# #s2
# # read file
# s2_dataset = pd.read_fwf('s2.txt', header=None, names=['x', 'y'])
#
# # dropping row with no data
# s2_dataset.apply(pd.to_numeric, errors='coerce').dropna()
#
# # ploting dataset
# x = s2_dataset['x']
# y = s2_dataset['y']
# plt.scatter(x, y)
# plt.show()
#
# # # finding wcss values for elbow method in order to find optimal number of clusters
# wcss = []
# K = range(5, 20)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k, init="k-means++", n_init=50, max_iter=12, random_state=42)
#     kmeanModel.fit(s2_dataset)
#     wcss.append(kmeanModel.inertia_)
#
# # ploting elbow
# plt.plot(K, wcss, 'bx-')
# plt.xlabel('number of clusters')
# plt.ylabel('Distortion')
# plt.title('Elbow method for s2')
# plt.show()
#
# optimal=15
# hsv_modified = cm.get_cmap('rainbow', optimal)
# kmeans = KMeans(n_clusters=optimal, init="k-means++", n_init=150, max_iter=13, random_state=42)
# kmeans.fit(s2_dataset)
# cluster_predict = kmeans.fit_predict(s2_dataset)
# cluster_centers = kmeans.cluster_centers_
# x2 = [x[0] for x in cluster_centers]
# y2 = [x[1] for x in cluster_centers]
# plt.scatter(x, y, cmap=hsv_modified, c=cluster_predict)
# plt.scatter(x2, y2, marker='*', s=500, c='k')
# plt.show()


# #s3
# # read file
# s3_dataset = pd.read_fwf('s3.txt', header=None, names=['x', 'y'])
#
# # dropping row with no data
# s3_dataset.apply(pd.to_numeric, errors='coerce').dropna()
#
# # ploting dataset
# x = s3_dataset['x']
# y = s3_dataset['y']
# plt.scatter(x, y)
# plt.show()
#
# # # finding wcss values for elbow method in order to find optimal number of clusters
# wcss = []
# K = range(5, 20)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k, init="k-means++", n_init=50, max_iter=12, random_state=42)
#     kmeanModel.fit(s3_dataset)
#     wcss.append(kmeanModel.inertia_)
#
# # ploting elbow
# plt.plot(K, wcss, 'bx-')
# plt.xlabel('number of clusters')
# plt.ylabel('Distortion')
# plt.title('Elbow method for s3')
# plt.show()
#
# optimal=15
# hsv_modified = cm.get_cmap('rainbow', optimal)
# kmeans = KMeans(n_clusters=optimal, init="k-means++", n_init=150, max_iter=13, random_state=42)
# kmeans.fit(s3_dataset)
# cluster_predict = kmeans.fit_predict(s3_dataset)
# cluster_centers = kmeans.cluster_centers_
# x2 = [x[0] for x in cluster_centers]
# y2 = [x[1] for x in cluster_centers]
# plt.scatter(x, y, cmap=hsv_modified, c=cluster_predict)
# plt.scatter(x2, y2, marker='*', s=500, c='k')
# plt.show()

# # #4
# # read file
# s4_dataset = pd.read_fwf('s4.txt', header=None, names=['x', 'y'])
#
# # dropping row with no data
# s4_dataset.apply(pd.to_numeric, errors='coerce').dropna()
# s4_dataset.dropna(axis=0)
#
# string_to_check = "it'son"
#
# for col in s4_dataset.columns:
#     s4_dataset = s4_dataset[~s4_dataset[col].astype(str).str.contains(string_to_check)]
#
#
# # ploting dataset
# x = s4_dataset['x']
# y = s4_dataset['y']
# plt.scatter(x, y)
# plt.show()
#
# # # finding wcss values for elbow method in order to find optimal number of clusters
# wcss = []
# K = range(5, 20)
# for k in K:
#     kmeanModel = KMeans(n_clusters=k, init="k-means++", n_init=50, max_iter=12, random_state=42)
#     kmeanModel.fit(s4_dataset)
#     wcss.append(kmeanModel.inertia_)
#
# # ploting elbow
# plt.plot(K, wcss, 'bx-')
# plt.xlabel('number of clusters')
# plt.ylabel('Distortion')
# plt.title('Elbow method for s4')
# plt.show()
#
# optimal=1
# hsv_modified = cm.get_cmap('rainbow', optimal)
# kmeans = KMeans(n_clusters=optimal, init="k-means++", n_init=150, max_iter=13, random_state=42)
# kmeans.fit(s4_dataset)
# cluster_predict = kmeans.fit_predict(s4_dataset)
# cluster_centers = kmeans.cluster_centers_
# x2 = [x[0] for x in cluster_centers]
# y2 = [x[1] for x in cluster_centers]
# plt.scatter(x, y, cmap=hsv_modified, c=cluster_predict)
# plt.scatter(x2, y2, marker='*', s=500, c='k')
# plt.show()


spiral_dataset = pd.read_csv(spiral_data_parser(), names=['x','y', 'z'])

# dropping row with no data
spiral_dataset.apply(pd.to_numeric, errors='coerce').dropna()
spiral_dataset.dropna(axis=0)

# ploting dataset
x = spiral_dataset['x']
y = spiral_dataset['y']
z = spiral_dataset['z']
plt.scatter(x, y)
plt.show()

# # finding wcss values for elbow method in order to find optimal number of clusters
wcss = []
K = range(5, 20)
for k in K:
    kmeanModel = KMeans(n_clusters=k, init="k-means++", n_init=50, max_iter=12, random_state=42)
    kmeanModel.fit(spiral_dataset)
    wcss.append(kmeanModel.inertia_)

# ploting elbow
plt.plot(K, wcss, 'bx-')
plt.xlabel('number of clusters')
plt.ylabel('Distortion')
plt.title('Elbow method for s4')
plt.show()

optimal=3
hsv_modified = cm.get_cmap('rainbow', optimal)
kmeans = KMeans(n_clusters=optimal, init="k-means++", n_init=150, max_iter=13, random_state=42)
kmeans.fit(spiral_dataset)
cluster_predict = kmeans.fit_predict(spiral_dataset)
cluster_centers = kmeans.cluster_centers_
x2 = [x[0] for x in cluster_centers]
y2 = [x[1] for x in cluster_centers]
plt.scatter(x, y, cmap=hsv_modified, c=cluster_predict)
plt.scatter(x2, y2, marker='*', s=500, c='k')
plt.show()

