import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


X = pd.read_csv("Mall_Customers.csv").iloc[:,3:5].values
# Sadece en önemlileri olan 4. ve 5. sütunları aldık çünkü ileride görselleştirirken anlaşılır olmasını istedik
# Gerçek bir problemle uğraşırken hepsini al.

"""
# K-Means modelinin kurulumu ve Optimum cluster sayısının tespiti
from sklearn.cluster import KMeans

wcss=[]
for i in range(1,11):
    kmc = KMeans(n_clusters=i,init="k-means++",random_state=0)
    kmc.fit(X)
    wcss.append(kmc.inertia_)
    # kmc.inertia_  komutu WCSS skorunu verir.
    
plt.plot(range(1,11),wcss)
plt.xlabel("Cluster Sayısı")
plt.ylabel("WCSS Skoru")
plt.title("Elbow Grafiği")
plt.show()
# Grafiği çizdirince anlarız ki Optimum cluster sayısı 5'tir.
"""

# Optimum cluster sayısıyla K-Means Modelinin kurulması
from sklearn.cluster import KMeans
kmc = KMeans(n_clusters=5,init="k-means++",random_state=0)
y_kmeans = kmc.fit_predict(X)

# Cluster'ların plot olarak çizilmesi
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100,c="blue",label="Class 1")
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100,c="red",label="Class 2")
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100,c="green",label="Class 3")
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100,c="purple",label="Class 4")
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=100,c="yellow",label="Class 5")
plt.scatter(kmc.cluster_centers_[:,0],kmc.cluster_centers_[:,1],s=225,c="black",label="Centroids")
plt.legend()
plt.title("Müşteri Segmentasyanu")
plt.xlabel("Gelir")
plt.ylabel("Harcama Skoru")
plt.show()
#print(kmc.cluster_centers_) # .cluster_centers_ kodu Centroidlerin koordinat bilgilerini verir.










