
# coding: utf-8

# In[1]:


#example of k-means clustering (unsupervised ML) on dataset of customers and income
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os

os.chdir(r"C:\Users\Zack\OneDrive\Documents\Part 4 - Clustering\Section 24 - K-Means Clustering\14_page_p4s24_file_1\K_Means") 
data_set = pd.read_csv('Mall_Customers.csv')
X = data_set.iloc[:, [3, 4]].values 
#Use elbow method to find optimal # of clusters 
from sklearn.cluster import KMeans 
wCSS = []
for i in range(1, 11): 
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0) #avoid random initialization error by using 'k-means++' 
    kmeans.fit(X) 
    wCSS.append(kmeans.inertia_)
    
#determine optimal # of clusters 
plt.plot(range(1, 11), wCSS)
plt.title('elbow method')
plt.xlabel('num of clusters')
plt.ylabel('wCSS')
plt.show()

#fit model 
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

#graph clusters 
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

