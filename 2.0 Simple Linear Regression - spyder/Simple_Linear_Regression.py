"""import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("deneyim-maas.csv",sep=";")
# sep parametresi defaultta ',' dür. Bizim verimiz ';' ile ayrıldığı için sep=";" yaptık.

plt.scatter(data.deneyim,data.maas)
# scatter verileri eksende nokta şeklinde görüntüler. (scatter = dağılım)
plt.xlabel("Deneyim")
plt.ylabel("Maaş")
plt.title("Deneyim-Maaş Grafiği")
plt.show()"""


# Satır başı yorumlar ana adımları temsil ediyor. Her makine öğrenmesi kodunda bunları uygula.

# Kütüphanelerin import edilmesi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Verilerin import edilmesi

dataset = pd.read_csv("deneyim-maas.csv",sep=";")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values   #Burda [:,:-1] ve [:,-1] yerine [:,0] ve [:,1] yazsaydın np.array 1 boyutlu olacağından veriyi LinearRegressiona sokarken reshape yapmamızı isteyecekti.


# Kayıp Dataların Düzeltilmesi
    # Veri Setimizde kayıp data olmadığı için gerek yok.



# Kategorik verilerin dönüştürülmesi
    # Veri setimiz sadece sayısal değerlerden oluşturulduğu için gerek yok.
    
    
# Train ve Test Setlerinin Oluşturulması

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)


# Feature Scaling (gerekliyse)
    # Uzaklık odaklı bir algoritma olmadığı için Feature Scaling'e gerek yok.
    
    
# Train Set Üzerinden Öğrenme
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)


# Modelin Test Set Üzerinde Denenmesi

y_predict = lr.predict(X_test)

# Train Set Sonuçlarının Görselleştirilmesi

plt.scatter(X_train,y_train,color="red")
plt.plot(X_train,lr.predict(X_train),color="blue")
plt.title("Deneyim-Maaş Grafiği")
plt.xlabel("Deneyim")
plt.ylabel("Maaş")
plt.show()


# Test Sonuçlarının Görselleştirilmesi

plt.scatter(X_test,y_test,color="red")
plt.plot(X_train,lr.predict(X_train),color="blue")
plt.title("Deneyim-Maaş Grafiği")
plt.xlabel("Deneyim")
plt.ylabel("Maaş") 
plt.show()












