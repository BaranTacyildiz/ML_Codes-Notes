import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Data.csv")
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)


#Polynomial Regression modelinin öğrenmesi
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
pol_reg = PolynomialFeatures(degree=4)
#Degree default 2.

X_pol = pol_reg.fit_transform(X_train)
lr2 = LinearRegression()
lr2.fit(X_pol,y_train)


# Tahmin Denemesi
y_pred = lr2.predict(pol_reg.transform(X_test)) # Zaten X_train ile fitlendi o yüzden sadece transform yazmamız yeterli.
ComparePolynomialRegression = np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),axis=1)
# axis=1 yaptık ki veri çift sütun olsun.

# R Squared Skoru
from sklearn.metrics import r2_score
R2ScorePolynomialRegression = r2_score(y_test,y_pred)

