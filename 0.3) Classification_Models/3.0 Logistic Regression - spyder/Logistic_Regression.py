import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Bilgisayar_Satis_Tahmin.csv")
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# Missing Value yok (büyük veri setlerinde kontrol etmeden impute yap.)
# Encodinge gerek yok. (Kategorik veri yok.)


# train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

# Feature Scaling'e gerek yok lakin yapılması faydalı olabilir.
# Bu notu okuduğun zaman Scaling yapılması ve yapılmaması arasındaki farka bak. (Sadece bilgilenme amaçlı)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
# y_train ve y_testi FeatureScaling'e tabi tutmayacağız çünkü zaten 0 ve 1 veriler.

# Logistic Regression modelinin kurulması
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

# Classifier tahmin denemesi
LogisticRegressionPredict = classifier.predict(ss.transform([[30,90000]]))

# 0 veya 1 yerine direkt olarak olasılık bilgisini almak
LogisticRegressionPredictProbability = classifier.predict_proba(ss.transform([[30,90000]]))

# Logistic Regression y_test tahmin denemesi
y_pred = classifier.predict(X_test)

# Confisuon Matrix (Hata Matrisi)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
AccuracyScore = accuracy_score(y_test, y_pred)


# Train set sonuçlarının görselleştirilmesi
from matplotlib.colors import ListedColormap
X_set = ss.inverse_transform(X_test)
# inverse_transform scale edilmiş veriyi eski haline döndürür.
y_set = y_test
X1, X2 = np.meshgrid(np.arange(start =X_set[:, 0].min() -10 , stop=X_set[:, 0].max() + 10, step=0.25),
                     np.arange(start =X_set[:, 1].min() -1000 , stop=X_set[:, 1].max() + 1000, step=0.25))
# np.meshgrid metodu; X1 x düzlemini, X2 y düzlemini temsil edecek şekilde bir koordinat düzlemi oluşturur.

plt.contourf(X1, X2, classifier.predict(ss.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('Logistic Regression - Test Set')
plt.xlabel('Yaş')
plt.ylabel('Maaş')
plt.legend()
plt.show()

# Train set için de görselleştirme yapabilirsin sadece 47. ve 49. satırlardaki testi train ile değiştir.


