#Implementation Of Complement Naive Bayes
#Importing Required Libraries And Modules
from sklearn.datasets import load_wine as wine
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report as classreport,accuracy_score as acc_score

#Loading The Dataset 
data=wine()
x,y=data.data,data.target

#Splittng The Dataset
x_train,x_test,y_train,y_test=ttsplit(x,y,
                                      test_size=0.3,random_state=42)
#Initializing The Classifier Model And Fitting The Data
cnb=ComplementNB()
cnb.fit(x_train,y_train)

#Making Prediction
pred=cnb.predict(x_test)
print("Accuracy:",acc_score(y_test,pred))
print("\nClassification Report:")
print(classreport(y_test,pred))

