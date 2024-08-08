import numpy as np
import pandas as pd
import matplotlib as plt

dataset = pd.read_csv("breast_cancer.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

# Özünde train edilmesi gerekmez ama kod gereği train_test_split işlemini gerçekleştirmemiz gerekiyor.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=1)

#DİSTANCE TEMELLİ BİR ALGORİTMA VE FEATURE SCALİNG YAPMAMIZ LAZIM.
# Feature Scaling (Özellik Ölçekleme)
# Standardization yöntemiyle kodlayacağız
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test= ss.transform(X_test)
# X_testte sadece transform işlemi yaptık çünkü X_trainde fit ile sahip olduğumuz ölçek X_test'te de geçerli olsun diye.

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)
y_predict = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cmKNN = confusion_matrix(y_test, y_predict)
acsKNN = accuracy_score(y_test, y_predict)
# Feature Scaling yapmasaydık ConfusionMatrix ve AccuracyScore gözle görülür şekilde düşük çıkacaktı.

# k-Fold Cross Validation modeli ile modelin performansının ölçülmesi
from sklearn.model_selection import cross_val_score
accuraciesKNN = cross_val_score(estimator=classifier,X=X,y=y,cv=10)
# estimator = yöntemimizi giriyoruz
# cv = kaç cross validation yapacağımızı giriyoruz.

# Standart Deviationa da bakarız çünkü standart deviation fazla ise hala bir bozukluk var demektir.(overfitting olabilir)
# Çünkü bir parçada %70 bir parçada %95 doğruluk oranı isabetli bir model oluşturmaz.
# Std az ise bir sorun yoktur.

k_foldKNN_validation_accuracy = accuraciesKNN.mean()*100
standart_deviationKNN = accuraciesKNN.std()*100
# Bu örnekte std az ve modelimizin doğruluk oranı yüksek.
