import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("Data.csv")

# Bağımsız ve Bağımlı değişkenleri ayırma
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
## Ara Not: Bilgi almak istediğin metodun sonuna tıkla ve CTRL+I yap

# Kayıp Data Doldurma

from sklearn.impute import SimpleImputer
# impute = atfetmek

imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
"""imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])"""
X[:,1:3] = imputer.fit_transform(X[:,1:3])

# imputer.fit ve imputer.transform işlevlerini kullanabilmek için X'in ndarray olması gerekiyor.

# Değişkenlerin "OneHotEncoder" methodu ile düzenlenmesi
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[0])],remainder="passthrough" )
X = ct.fit_transform(X)
# X = np.array(ct.fit_transform(X))ColumntTransformer.fit_transform np array döndürmezse böyle yaz.

# Değişkenlerin "LabelEncoder" methodu ile düzenlenmesi
# LabelEncoding 2 seçenek olduğu durumlarda kullanılır (0/1), (yes/no) etc.
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

# Train ve Test setleri

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)
