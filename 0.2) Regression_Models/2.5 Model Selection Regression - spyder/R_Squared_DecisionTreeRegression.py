import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Data.csv")
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 1)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
# random_state = 0 dememizin sebebi kurs ile aynı traini kullanıp aynı sonucu almak.
regressor.fit(X_train,y_train)

# Tahmin Denemesi
y_pred = regressor.predict(X_test)
CompareDecisionTreeRegression = np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),axis=1)
# axis=1 yaptık ki veri çift sütun olsun.

# R Squared Skoru
from sklearn.metrics import r2_score
R2ScoreDecisionTreeRegression = r2_score(y_test,y_pred)




