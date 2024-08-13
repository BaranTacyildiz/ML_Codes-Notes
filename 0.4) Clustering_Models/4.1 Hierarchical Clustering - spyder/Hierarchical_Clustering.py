import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


X = pd.read_csv("Mall_Customers.csv").iloc[:,3:5].values
# Bilerek 3:5 aralığını kullanıyoruz. Modeli daha iyi anlamak için.
# 2 Değişkenli yapacağız ki görüntüleyince iyi anlayalım.

# Optimum Cluster sayısını bulmak için Dendrogram kullanımı
"""import scipy.cluster.hierarchy as sch

dendrgram = sch.dendrogram(sch.linkage(dataset,method="ward"))
# İyi bir sonuç için method="ward" isimli yöntem tercih edilir.

plt.title("Dendogram Ward")
plt.xlabel("Customers")
plt.ylabel("Distance")
"""

# Hierarchical Clustering Agglomerative modelinin kurulması
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=5,metric="euclidean",linkage="ward")
#  If linkage is "ward", only "euclidean" is accepted for metric parameter.
y_ac = ac.fit_predict(X)

plt.scatter(X[y_ac==0,0],X[y_ac==0,1],s=100,c="blue",label="Class 1")
plt.scatter(X[y_ac==1,0],X[y_ac==1,1],s=100,c="red",label="Class 2")
plt.scatter(X[y_ac==2,0],X[y_ac==2,1],s=100,c="green",label="Class 3")
plt.scatter(X[y_ac==3,0],X[y_ac==3,1],s=100,c="purple",label="Class 4")
plt.scatter(X[y_ac==4,0],X[y_ac==4,1],s=100,c="yellow",label="Class 5")
plt.legend()
plt.title("Müşteri Segmentasyanu")
plt.xlabel("Gelir")
plt.ylabel("Harcama Skoru")
plt.show()





