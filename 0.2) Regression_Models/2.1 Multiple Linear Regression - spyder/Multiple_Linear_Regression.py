# Kütüphanelerin import edilmesi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Verilerin import edilmesi

data = pd.read_csv("50_Startups.csv")
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# Missing Valueların Tanımlanması
# Value'nin 0 olması missing olduğu anlamına gelmez. 
# Bu nedenle bu verisetinde kayıp data tanımlaması yapmamız gerekmiyor.
# Veriseti çok büyükse (gözle kontrol edilmeyecek kadar)
# Missing Value kodunu yaz missing value varsa tamamlar.


# Değişkenlerin OneHotEncoding yöntemi ile düzenlenmesi

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[3])],remainder="passthrough" )
X = ct.fit_transform(X)

# Feature Scaling (Özellik Ölçekleme) kullanacağımız algoritma uzaklık temelli olmadığı için scaler kullanmaya gerek yok.
# coefficent değerleri zaten scaling yapmış gibi davranıyor.


# Train-Test Setlerini ayırmak

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)

# Multiple Linear Regression modelinin Train set üzerinden öğrenmesi

from sklearn.linear_model import LinearRegression

mlr = LinearRegression()
mlr.fit(X_train,y_train)

y_predict = mlr.predict(X_test)


# Bunun görselleştirilmesi üzerine çalış

plt.scatter(y_test, y_predict, color='blue', label='Gözlemler')
plt.plot(y_test, y_test, color='red',label='Doğru Tahmin')
plt.title('Gerçek Değerler ve Tahminler')
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahminler')
plt.legend()
plt.show()


# Modelin Test Edilmesi
# -> OneHotEncoding metodunun kullanımıyla ortaya çıkan Dummy Variable Trap durumu nedir? Nasıl Çözülür? araştır.
# Udemy Multiple Linear Regression son videosunda var.




