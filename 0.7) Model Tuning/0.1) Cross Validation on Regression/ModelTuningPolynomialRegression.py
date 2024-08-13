# -*- coding: utf-8 -*-

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



#Polynomial (Linear) Regression Modelinin Öğrenmesi
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
pol_reg = PolynomialFeatures(degree=4)

X_pol = pol_reg.fit_transform(X_train)

lr2 = LinearRegression()
lr2.fit(X_pol, y_train)

#Tahmin Denemesi
y_pred = lr2.predict(pol_reg.transform(X_test))
ComparePolynomialRegression = np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), axis = 1)

#R Squared Skoru
from sklearn.metrics import r2_score
R2ScorePolynomialRegression = r2_score(y_test, y_pred)


#k-Fold Cross Validation Yöntemiyle Modelin Preformansının Ölçülmesi
from sklearn.model_selection import cross_val_score
RSquares = cross_val_score(estimator = lr2, X = X, y = y, cv = 10, scoring="r2")
RSquaresMeanPR = RSquares.mean()*100
StandartDeviation = RSquares.std()*100














