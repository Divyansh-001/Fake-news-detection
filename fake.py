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

tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)

X = tfidf.fit_transform(x)

pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(X, y)


print("number of news to check")
n = int(input())
x_test = []
for i in range(0, n):
    z = input()
    x_test.append(z)

x_test = np.array(x_test)
x_test.resize(3)

X_test = tfidf.transform(x_test)
y_pridict = pac.predict(X_test)