import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

dataset = pd.read_csv('news.csv')
x = dataset.iloc[:, 2].values
y = dataset.iloc[:, 3].values


x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)

X_train = tfidf.fit_transform(x_train)
X_test = tfidf.transform(x_test)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(X_train, y_train)

y_predict = pac.predict(X_test)
score=accuracy_score(y_test,y_predict)
