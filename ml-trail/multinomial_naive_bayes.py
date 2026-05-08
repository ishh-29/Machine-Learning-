#Implementation Of Multinomial Naive Bayes
#Importing Required Modules And Libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#Creating The Dataset
data={
    'text':[
        'Free money now',
        'Call now to claim your prize',
        'Meet me at the park',
        'Let’s catch up later',
        'Win a new car today!',
        'Lunch plans?',
        'Congratulations! You won a lottery',
        'Can you send me the report?',
        'Exclusive offer for you',
        'Are you coming to the meeting?'
    ],
    'label':['spam','spam','not spam','not spam','spam','not spam','spam','not spam','spam','not spam']
}
df=pd.DataFrame(data)

#Mapping Lables To Numerical Values
df['label']=df['label'].map({'spam':1,'not spam':0})

#Splitting The Data
x=df['text']
y=df['label']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#Vectorizing The Text Data
'''
CountVectorizer->Used To Convert Text Data Into Numerical 
                 Vectors. Counts The Frequency Of Each Word In 
                 The Corpus.
fit_transform()->For The Training Data To Learn Vocabulary
                 And Transform It Into A Feature Matrix
'''
vrizer=CountVectorizer()
train_vec=vrizer.fit_transform(x_train)
test_vec=vrizer.transform(x_test)

#Training The Model
model=MultinomialNB()
model.fit(train_vec,y_train)

#Making Predictions And Calculating Accuracy
pred=model.predict(test_vec)
acc=accuracy_score(y_test,pred)
print(f"Accuracy:{acc*100:.2f}%\n")

#Predicting For A Custom Message
message=["Congratulations, you've won a free vacation"]
print(message)
custom_vec=vrizer.transform(message)
prediction=model.predict(custom_vec)
print("Prediction For Custom Message:","Spam"if prediction[0]==1 else "Not Spam")