import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("kalite-fiyat.csv")
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

"""
- Missing Value yok (Büyük veride bilmesen de impute yap)
- Encodinge gerek yok
- Feature Scalinge gerek yok 
- Veri Seti çok küçük olduğu için train_test_split yapmayacağız.
"""

# Decision Tree modelinin eğitilmesi
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
# random_state = 0 dememizin sebebi kurs ile aynı traini kullanıp aynı sonucu almak.
regressor.fit(X,y)

# Decision Tree modeli ile tahmin denemesi
y_predict = regressor.predict(np.array([[6.5]]))

# Bu verisetinde DecisionTree algorithm kötü bir sonuç sergiler.
# DecisionTree algorithm fazla feature içeren verisetlerinde daha başarılıdır.

# DecisionTree modelinin görselleştirilmesi
"""
plt.scatter(X,y)
plt.plot(X,regressor.predict(X),color="red")
plt.title("Decision Tree Modeli")
plt.xlabel("Kalite")
plt.ylabel("Fiyat")
plt.show()
"""

# Çizdirilen grafik oturmuş gibi gözükse de kırılım sayısını arttırırsak asıl kırılımı görürüz.


X_grid = np.arange(min(X[:, 0]), max(X[:, 0]), 0.1)
X_grid = X_grid.reshape((len(X_grid),1)) # Başta X'te 1 satır, 90 sütun vardı
# Reshape metodunu kullanarak 90 satır 1 sütun haline çevirdik veriyi.


plt.scatter(X,y)
plt.plot(X_grid,regressor.predict(X_grid),color="red")
plt.title("Decision Tree Modeli")
plt.xlabel("Kalite")
plt.ylabel("Fiyat")
plt.show()





