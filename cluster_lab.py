from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt

# KMeans
np.random.seed(2)
x = np.random.randn(50,2)
x[:25,0] += 3
x[:25,1] -= 4

kmeans = KMeans(n_clusters=2, n_init=20).fit(x)
print(kmeans.labels_)
plt.scatter(x[:,0], x[:,1], c=kmeans.labels_)
plt.title("K-Means Clustering Results with K=2")
plt.show()

# Repeat for K=3 with different seed
np.random.seed(4)
kmeans = KMeans(n_clusters=3, n_init=20).fit(x)
plt.scatter(x[:,0], x[:,1], c=kmeans.labels_)
plt.title("K-Means Clustering Results with K=3")
plt.show()

# Hierarchical clustering
hc_complete = linkage(x, method='complete')
hc_average = linkage(x, method='average')
hc_single = linkage(x, method='single')

plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.6)

plt.subplot(1, 3, 1)
dendrogram(hc_complete)
plt.title("Complete Linkage")

plt.subplot(1, 3, 2)
dendrogram(hc_average)
plt.title("Average Linkage")

plt.subplot(1, 3, 3)
dendrogram(hc_single)
plt.title("Single Linkage")

plt.show()
