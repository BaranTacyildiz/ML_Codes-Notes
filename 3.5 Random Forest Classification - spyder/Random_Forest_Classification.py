import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Bilgisayar_Satis_Tahmin.csv")
# Bağımsız ve Bağımlı değişkenleri ayırma
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
 
# Train ve Test setleri
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)

# Feature Scaling (Özellik Ölçekleme)
# Algoritma uzaklık temelli olmasa da tahmin kalitesini iyileştirmek için Feature Scaling yapabiliriz.
# Standardization yöntemiyle kodlayacağız
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

# Random Forest modelinin kurulması
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,random_state=0,criterion="entropy")

# Modelin X_train ile eğitilmesi
classifier.fit(X_train,y_train)

# Modelin X_test üzerinde denenmesi
y_predict = classifier.predict(X_test)

# Confusion Matrix ve Accuracy Score Değerleri
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_predict)
acs = accuracy_score(y_test, y_predict)
# Feature Scaling yapmasaydık ConfusionMatrix ve AccuracyScore gözle görülür şekilde düşük çıkacaktı.


