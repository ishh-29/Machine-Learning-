#Implementing Linear SVM Algorithm
#Importing Required Modules And Libraries
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import SVC

#Loading The Dataset
cancer=load_breast_cancer()
x=cancer.data[:,:2]
y=cancer.target

#Initializing SVM 
svm=SVC(kernel="linear",C=1)
svm.fit(x,y)

#Plotting The Model
DecisionBoundaryDisplay.from_estimator(
    svm,
    x,
    response_method="predict",
    alpha=0.8,
    cmap="Pastel1",
    xlabel=cancer.feature_names[0],
    ylabel=cancer.feature_names[1],
    )
plt.scatter(x[:,0],x[:,1],c=y,s=20,edgecolors="k")
plt.show()