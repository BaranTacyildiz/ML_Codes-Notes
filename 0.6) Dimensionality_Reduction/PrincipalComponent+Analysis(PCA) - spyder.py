import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("breast_cancer.csv")

# Bağımsız ve Bağımlı değişkenleri ayırma
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# Train ve Test setleri
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)


# Feature Scaling (Özellik Ölçekleme)
# Standardization yöntemiyle kodlayacağız
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train[:,:] = ss.fit_transform(X_train[:,:])
X_test[:,:] = ss.transform(X_test[:,:])
# X_testte sadece transform işlemi yaptık çünkü X_trainde fit ile sahip olduğumuz ölçek X_test'te de geçerli olsun diye.
# direkt X_train yerine X_train[:,:] yazdığımız için standardizasyon sonucu farklı çıktı. Kontrol et.
# böyle yapmamız PCA'yı da etkiliyor.

# PCA Modelinin kurulması
from sklearn.decomposition import PCA
pca = PCA(n_components=2,random_state=0)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Varyans ne kadar korundu ?
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))
# ilk feature'nin 0.49 ikinci feature'nin 0.12 varyansı korunmuştur.
# toplamda 0.61 yani %61 korundu varyans.
# pca = PCA(n_components=) değeriyle oynayarak varyans korunumunu değiştirebilirsin.

# Logistic Regression modelinin kurulması
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

# Confisuon Matrix (Hata Matrisi) ve AccuracyScore
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
AccuracyScore = accuracy_score(y_test, y_pred)



# Train set sonuçlarının görselleştirilmesi
from matplotlib.colors import ListedColormap
X_set = X_train
y_set = y_train
X1, X2 = np.meshgrid(np.arange(start =X_set[:, 0].min() -10 , stop=X_set[:, 0].max() + 10, step=0.25),
                     np.arange(start =X_set[:, 1].min() -10 , stop=X_set[:, 1].max() + 10, step=0.25))
# np.meshgrid metodu; X1 x düzlemini, X2 y düzlemini temsil edecek şekilde bir koordinat düzlemi oluşturur.

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'blue'))(i), label = j)
plt.title('Logistic Regression - PCA')
plt.xlabel('P1')
plt.ylabel('P2')
plt.legend()
plt.show()

# Train set için de görselleştirme yapabilirsin sadece 47. ve 49. satırlardaki testi train ile değiştir.




