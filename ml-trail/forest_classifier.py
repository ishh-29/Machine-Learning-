#Implementation Of Random Forest Classification 
#Importing Required Modules And Libraries 
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score as acc_score, confusion_matrix as con_mat
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

#Loading The Dataset
iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target

#Preparing Data
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

#Splitting The Dataset
x_train,x_test,y_train,y_test=ttsplit(x,y,
                                      test_size=0.2,random_state=42)

#Feature Scaling 
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#Initializing The Classifier Model
clsfier=RandomForestClassifier(n_estimators=100,random_state=42)
clsfier.fit(x_train,y_train)
pred=clsfier.predict(x_test)

#Calculating Accuracy
acc=acc_score(y_test,pred)
print(f'Accuracy:{acc*100:.2f}%')
mat=con_mat(y_test,pred)

#Plotting The Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(con_mat,annot=True,
            fmt='g',cmap='Blues',
            cbar=False,xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

#Plotting Importantance Of Features
feat_imp=clsfier.feature_importances_
plt.barh(iris.feature_names,feat_imp)
plt.xlabel('Feature Importance')
plt.title('Feature Importance In Random Forest Classification')
plt.show()
