import pandas as pd


dataset = pd.read_csv("Restaurant_Reviews.tsv",sep = "\t",quoting=3)


import re
# re => regular expressions
# re.sub => substition yani yerine koymak 

# 1) Cleaning the Data
def preprocess_review(review):
    return re.sub(r'[^a-zA-Z]', ' ', review).lower().split()

    # Preprocess_review fonksiyonu elimizdeki verinin satırlarındaki latin harfleri hariç karakterleri siler, kelimelerin hepsini küçük
    # harflerle yazar (sekans bozulmasını önlemek için) ve tüm cümlelerin kelimelerini liste şeklinde ayırır. (Bu da sekans ölçümü için)

# Apply fonksiyonu ile fonksiyonun Review sütununa uygulanması
dataset['Review'] = dataset['Review'].apply(preprocess_review)

# 2) Stopword Cleaning
import nltk
#nltk.download("stopwords")
from nltk.corpus import stopwords


all_stopwords = stopwords.words("english")
all_stopwords.remove("not")

for i in range(len(dataset)):  # Tüm veriyi dolaş
    filtered_words = [word for word in dataset["Review"][i] if word not in all_stopwords]
    dataset["Review"][i] = filtered_words


# Dikkat edilmesi gereken bir husus NLP uygulamalarında olumsuz cümleyi anlamak meşakatli bir işlemdir.
# Örneğin bu uygulamada not gibi kelimeleri çıkardığımız için olumsuz yargıları ayırmak zorlaşacaktır.
# .remove("not") ile sadece not kelimesinin bulunduğu cümlelerde bunun önüne geçmeye çalıştık.


# 3) Stemming
# stemming adımında kelimelerin köklerini tespit etmeye çalışacağız.
# waited,waiting,waits -> wait
# stemming yönteminden başka kök bulma yöntemleri de vardır.

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

for i in range(len(dataset)):  # Tüm veriyi dolaş
    # Stemming işlemi
    stemmed_words = [ps.stem(word) for word in dataset["Review"][i]]
    dataset["Review"][i] = stemmed_words
    


# 4) Bag of Words
# Bag of Words'ü uygulamadan önce cümlelerimizi liste halinden tekrar tek cümle haline çevirmemiz gerekiyor.
   
for i in range(len(dataset)):
    joined = (" ".join(dataset["Review"][i]))
    dataset["Review"][i] = joined

# Bag of words kelimelerin sekansını belirleyen yöntemdir.

from sklearn.feature_extraction.text import CountVectorizer
# extraction = özünü alma. özünü çıkarma.
cv = CountVectorizer(max_features=1500)
# max features değerini X'i bulup sütun sayısına baktıktan sonra verdik. Önce değil! Sen de öyle yap. 
# max_features'i az kullanılak özel isim vb kelimeleri çıkarmak için X'in sütun sayısından yaklaşık 60 eksik verdik.
X = cv.fit_transform(dataset["Review"]).toarray()
y = dataset.iloc[:,-1].values


# NLP modelinin kurulması
# NLP classification'a benzediği için burada NLP için Naive Bayes yöntemini kullacanağız.

# Train ve Test setleri
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)
# Naive Bayes modelinin kurulması
# internete Sklearn naive bayes yazıp sklearn sitesine girersek birden fazla NaiveBayes yöntemi olduğunu görürüz.
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

# Modelin Eğitilmesi
classifier.fit(X_train,y_train)


#* Modelin Test Set üzerinde tahmini
y_predict = classifier.predict(X_test)

# Confusion Matrix ve Accuracy Score Değerleri
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_predict)
acs = accuracy_score(y_test, y_predict)


# Başka classification yöntemleri deneyerek de test edebilirsin. acNB = 0.685 acSVM(rbf) = 0.825

"""
# Support Vector Machine modelinin kurulması
from sklearn.svm import SVC
classifier = SVC(probability=True,kernel="rbf") # Kernel = rbf,linear,poly,sigmoid |||| linear kernel Logistic Regression'a benzer.
# kernel = svm modelinin hangi algoritma ile kurulacağını belirliyor. acs'yi etkiler
# 33. satırdaki .predict_proba fonksiyonunun çalışması için SVC içinde probability'i True yapmalıyız.

# Modelin eğitilmesi
classifier.fit(X_train,y_train)

# Tahmin denemesi
# Test set tahmini
y_predict = classifier.predict(X_test)
y_predict_proba = classifier.predict_proba(X_test)


# Confusion Matrix ve Accuracy Score Değerleri
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_predict)
acs = accuracy_score(y_test, y_predict)
"""











    
            

