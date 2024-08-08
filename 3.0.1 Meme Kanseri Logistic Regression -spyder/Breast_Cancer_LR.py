import numpy as np
import pandas as pd
import matplotlib as plt

dataset = pd.read_csv("breast_cancer.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)

y_predict = classifier.predict(X_test)
y_predict_proba = classifier.predict_proba(X_test)


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_predict)
acs = accuracy_score(y_test,y_predict)

# k-Fold Cross Validation modeli ile modelin performansının ölçülmesi
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier,X=X,y=y,cv=10)
# estimator = yöntemimizi giriyoruz
# cv = kaç cross validation yapacağımızı giriyoruz.

# Standart Deviationa da bakarız çünkü standart deviation fazla ise hala bir bozukluk var demektir.(overfitting olabilir)
# Çünkü bir parçada %70 bir parçada %95 doğruluk oranı isabetli bir model oluşturmaz.
# Std az ise bir sorun yoktur.

k_fold_validation_accuracy = accuracies.mean()*100
standart_deviation = accuracies.std()*100
# Bu örnekte std az ve modelimizin doğruluk oranı yüksek.


