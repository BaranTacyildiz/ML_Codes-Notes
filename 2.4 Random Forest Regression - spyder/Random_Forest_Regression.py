import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("kalite-fiyat.csv")
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# Missing Value impute^'a gerek yok
# Encoding'e gerek yok
# Feature Scaling' gerek yok
# Veri seti küçük olduğu için train_test_split yapmayacağız.
# ( Normal şartlarda train_test_split'i yap)

# Random Forest Modelinin eğitilmesi
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(random_state=0,n_estimators = 10)
# n_estimators parametresi kaç tane DecisionTree ile RandomForest oluşturacağımızı belirler.
regressor.fit(X,y)

# Tahmin denemesi
y_predict_random_forest = regressor.predict(np.array([[6.5]]))

# RandomForest modeliminizin görselleştirilmesi

X_grid = np.arange(min(X[:, 0]), max(X[:, 0]), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))# Başta X'te 1 satır, 90 sütun vardı
# Reshape metodunu kullanarak 90 satır 1 sütun haline çevirdik veriyi.
plt.scatter(X,y)
plt.plot(X_grid,regressor.predict(X_grid),color ="red")
plt.title("Random Forest Modeli")
plt.xlabel("Kalite")
plt.ylabel("Fiyat")
plt.show()
