import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Bilgisayar_Satis_Tahmin.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)

# Distance temelli bir algoritma olduğu için Scaling yaptık.
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# Support Vector Machine modelinin kurulması
from sklearn.svm import SVC
classifier = SVC(probability=True,kernel="linear") # Kernel = rbf,linear,poly,sigmoid |||| linear kernel Logistic Regression'a benzer.
# kernel = svm modelinin hangi algoritma ile kurulacağını belirliyor. acs'yi etkiler
# 33. satırdaki .predict_proba fonksiyonunun çalışması için SVC içinde probability'i True yapmalıyız.

# Modelin eğitilmesi
classifier.fit(X_train,y_train)

# Tahmin denemesi
SVM_predict = classifier.predict(ss.transform([[25,80000]]))
# Test set tahmini
y_predict = classifier.predict(X_test)
y_predict_proba = classifier.predict_proba(X_test)


# Confusion Matrix ve Accuracy Score Değerleri
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_predict)
acs = accuracy_score(y_test, y_predict)
# Feature Scaling yapmasaydık ConfusionMatrix ve AccuracyScore gözle görülür şekilde düşük çıkacaktı.

"""
#Train Set Sonuçlarının Görselleştirilmesi ( Kopyala-Yapıştır yapıldı ayrıca çalış.)
from matplotlib.colors import ListedColormap
X_set = ss.inverse_transform(X_train)
y_set = y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.25),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.25))

plt.contourf(X1, X2, classifier.predict(ss.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('red', 'blue'))(i), label=j)
plt.title('Support Vector Machine - Train Set (RBF Kernel)')
plt.xlabel('Yaş')
plt.ylabel('Maaş')
plt.legend()
plt.show()
"""