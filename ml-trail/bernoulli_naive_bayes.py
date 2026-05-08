#Implementing Bernoulli Naive Bayes
#Importing Required Modules And Libraries
import numpy as np, pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report as report

#Reading The Dataset
df=pd.read_csv("spam_ham_dataset.csv")
print(df.shape)
print(df.columns)
df=df.drop(['Unnamed: 0'],axis=1)

#Count Vectorizer
x=df["text"].values
y=df["label_num"].values
cv=CountVectorizer()
x=cv.fit_transform(x)

#Splitting The Data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

#Training The Model
bnb=BernoulliNB(binarize=0.0)
model=bnb.fit(x_train,y_train)

#Making Prediction
pred=bnb.predict(x_test)
print(report(y_test,pred))
