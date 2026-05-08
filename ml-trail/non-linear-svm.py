#Implementing Non-Linear SVM
#With Circular Decision Boundary
#Importing Required Modules And Libraries
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_circles,make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

#Creating Datasets
x,y=make_circles(n_samples=500,factor=0.5,
                noise=0.05,random_state=42)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#Initializing Model
svm=SVC(kernel='rbf',C=1,gamma=0.5)
svm.fit(x_train,y_train)

#Making Predictions
pred=svm.predict(x_test)
acc=accuracy_score(y_test,pred)
print(f"Accuracy:{acc:.2f}")

#Visualizing Decision Boundnary 
def plot_decision(x,y,model):
    min_x,max_x=x[:,0].min()-1,x[:,0].max()+1
    min_y,max_y=x[:,1].min()-1,x[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(min_x,max_x,0.01),
                      np.arange(min_y,max_y,0.01))
    z=model.predict(np.c_[xx.ravel(),yy.ravel()])
    z=z.reshape(xx.shape)
    #Plotting
    plt.contour(xx,yy,z,alpha=0.8,cmap=plt.cm.Paired)
    plt.scatter(x[:,0],x[:,1],c=y,edgecolors='k',cmap=plt.cm.Paired)
    plt.title("Non-Linear SVM With RBF Kernel")
    plt.show()

plot_decision(x,y,svm)

#With Radial Curve Pattern
#Creating And Splitting The Dataset
x,y=make_moons(n_samples=500,noise=0.1,random_state=42)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#Creating And Training The Model With Polynomial Kernel
svm_poly=SVC(kernel='poly',degree=3,C=1,coef0=1)
svm_poly.fit(x_train,y_train)

#Making Predictions And Evaluating The Model
pred=svm_poly.predict(x_test)
acc=accuracy_score(y_test,pred)
print(f"Accuracy:{acc:.2f}")

#Visualizing The Decision Boundary
def decision_boundary(x,y,model):
    min_x,max_x=x[:,0].min()-1,x[:,0].max()+1
    min_y,max_y=x[:,1].min()-1,x[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(min_x,max_x,0.01),
                      np.arange(min_y,max_y,0.01))
    z=model.predict(np.c_[xx.ravel(),yy.ravel()])
    z=z.reshape(xx.shape)
    #Plotting
    plt.contourf(xx,yy,z,alpha=0.8,cmap=plt.cm.Paired)
    plt.scatter(x[:,0],x[:,1],c=y,edgecolors='k',cmap=plt.cm.Paired)
    plt.title("Non-Linear SVM With Polynomial Kernel")
    plt.show()

decision_boundary(x,y,svm_poly)