import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Veri setinin alınması, dependent, independent variable ayrımı.

data = pd.read_csv("kalite-fiyat.csv")
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

# Missing value impute edilmesien gerek yok

# Encodinge gerek yok.

# Train test split ayırmayacağız çünkü veri seti çok küçük. ayırırsak regresyon bozulur.
# Polynomial regression yaparken verisetin büyük olunca train test spliti yap. Bunda yapmayacağız sadece.

# Feature Scalinge gerek yok.

# Linear Regression modelinin öğrenmesi
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)


#Polynomial Regression modelinin öğrenmesi
from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures(degree=6)
#Degree overfitting olmadığı sürece arttıkça doğruluğu arttırır.
#Default 2 zaten.

X_pol = pol_reg.fit_transform(X)
lr2 = LinearRegression()
lr2.fit(X_pol,y)

# Linear modelin görselleştirilmesi
plt.scatter(X,y)
plt.plot(X,lr.predict(X),color="red")
plt.title("Kalite-Fiyat Grafiği (Linear)")
plt.xlabel("Kalite")
plt.ylabel("Fiyat")
plt.show()

# Polynomial modelin görselleştirilmesi
plt.scatter(X,y)
plt.plot(X,lr2.predict(X_pol),color="red")
plt.title("Fiyat-Kalite Grafiği (Polynomial) Degree = 6")
# Bu polynomial regressionda degree 4-8 aralığı idealdir. (deneyerek bulduk.)
plt.show()


# Linear modelin tahmin denemesi
linear_predict = lr.predict(np.array([[8.5]]))

# Polynomial modelin tahmin denemesi
polynomial_predict = lr2.predict(pol_reg.fit_transform([[10]]))










