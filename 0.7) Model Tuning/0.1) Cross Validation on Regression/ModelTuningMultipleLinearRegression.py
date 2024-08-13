import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Verilerin Import Edilmesi
dataset = pd.read_csv("Data.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Train ve Test Setleri
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#Multiple Linear Regression Modelinin Train Set Üzerinde Öğrenmesi
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Tahmin Denemesi
y_pred = regressor.predict(X_test)
CompareMultipleLinearRegression = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1)

#R Squared Skoru
from sklearn.metrics import r2_score
R2ScoredMultipleLinear = r2_score(y_test, y_pred)


#k-Fold Cross Validation Yöntemiyle Modelin Preformansının Ölçülmesi
from sklearn.model_selection import cross_val_score
RSquares = cross_val_score(estimator = regressor, X = X, y = y, cv = 10, scoring="r2")
RSquaresMeanMLR = RSquares.mean()*100
StandartDeviation = RSquares.std()*100


























