#Implementing Gaussian Naive Bayes
#Importing Required Modules And Libraries
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm

#Loading The Dataset
iris=load_iris()
data=pd.DataFrame(iris.data,columns=iris.feature_names)
data['Species']=iris.target
x=data.drop("Species",axis=1)
y=data['Species']

#Encoding And Splitting The Dataset
l_encode=LabelEncoder()
y=l_encode.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#Training The Model
gnb=GaussianNB()
gnb.fit(x_train,y_train)

#Plotting 1D Gaussian Distributions For All Features
feat_names=iris.feature_names
num_feat=len(feat_names)
num_class=len(np.unique(y))
npx=x.to_numpy()

for i in range(num_feat):
    feature=feat_names[i]
    valx=np.linspace(npx[:,i].min(),npx[:,i].max(),200)

    plt.figure(figsize=(8,4))
    
    for j in range(num_class):
        mean=gnb.theta_[j,i]
        std=np.sqrt(gnb.var_[j,i])
        valy=norm.pdf(valx,mean,std)
        #Plotting
        plt.plot(valx,valy,label=f"Class {j}({iris.target_names[j]})")
        plt.title(f"Gaussian Distribution->{feature}")
        plt.xlabel(feature)
        plt.ylabel("Probability Density")
        plt.legend()
        plt.grid(True)
        plt.show()

#Making Predictions
pred=gnb.predict(x_test)
acc=accuracy_score(y_test,pred)
print(f"Accuracy On Iris Flower:{acc}")
