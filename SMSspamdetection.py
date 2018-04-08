# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 02:14:53 2018

@author: Jay
"""

import numpy as np
import pandas as pd

dataset=pd.read_csv('SMSspamCollection', sep='\t', names=['lable', 'messages'])

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[] # initialize empty corpus
for i in range(0,5572):
    review=dataset['messages'][i]
    review=re.sub('[^a-zA-Z]',' ', review) # use regular expression to remove everythin except the range of words from a-z and A-z, separate them by a space.
    review=review.lower() # convert to lower case
    review=review.split() # covert to a list
    ps=PorterStemmer() 
    review= [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review) # covert to string
    corpus.append(review)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()
le=LabelEncoder()
y=le.fit_transform(dataset.iloc[:,0])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

