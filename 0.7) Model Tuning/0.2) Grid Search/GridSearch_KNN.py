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
classifier = KNeighborsClassifier()


# Grid Search
from sklearn.model_selection import GridSearchCV
# CV = Cross Validation

# GridSearch için hiperparametreler
hyper_params = {"n_neighbors": np.arange(1, 50)}
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=hyper_params,
                           scoring="accuracy",  
                           n_jobs=-1, # n_jobs kaç işlemci kullanacağını belirtir. -1 hepsi demektir.
                           cv=10) # Cross Validation adeti

# Grid Search'i uygula
grid_search.fit(X_train, y_train)

bestScore = grid_search.best_score_  
bestParam = grid_search.best_params_

print(f"Best Score: {bestScore}")
print(f"Best Parameters: {bestParam}")

