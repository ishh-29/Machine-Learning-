'''
Feature Engineering:
'''
#Importing The Modules
import numpy as np,matplotlib.pyplot as plt
from lab_utils_multi import zscore_normalize_features, run_gradient_descent_feng
np.set_printoptions(precision=2)  #Reduced Display Precision On Numpy Arrays

#Creating Target Data
x=np.arange(0,20,1)
y=1+x**2
X=x.reshape(-1,1)
model_w,model_b=run_gradient_descent_feng(X,y,iterations=1000,alpha=1e-2)
plt.scatter(x,y,marker='x',c='r',label="Actual Value");plt.title("No Feature Engineering")
plt.plot(x,X@model_w+model_b,label="Predicted Value");plt.xlabel("X");plt.ylabel("y");plt.legend()
plt.show()

#Engineered Feature
X=x**2
X=X.reshape(-1,1)
model_w,model_b=run_gradient_descent_feng(X,y,iterations=10000,alpha=1e-5)
plt.scatter(x,y,marker='x',c='r',label="Actual Value");plt.title("Feature Engineering")
plt.plot(x,np.dot(X,model_w)+model_b,label="Predicted Value");plt.xlabel("x");plt.ylabel("y");plt.legend()
plt.show()

X=np.c_[x,x**2,x**3]
model_w,model_b=run_gradient_descent_feng(X, y, iterations=10000, alpha=1e-7)
plt.scatter(x,y,marker='x',c='r',label="Actual Value");plt.title("x,x**2,x**3 Features")
plt.plot(x,X@model_w+model_b,label="Predicted Value");plt.xlabel("x");plt.ylabel("y");plt.legend()
plt.show()

#Via Scaling Features
X=np.c_[x,x**2,x**3]
print(f"Peak To Peak Range By Column In Raw X:{np.ptp(X,axis=0)}")
#Adding Mean Normalization
X=zscore_normalize_features(X)
print(f"Peak To Peak Range By Column In Normalized X:{np.ptp(X,axis=0)}")
model_w,model_b=run_gradient_descent_feng(X,y,iterations=100000,alpha=1e-1)
plt.scatter(x,y,marker='x',c='r',label="Actual Value");plt.title("Normalized x,x**2,x**3 Feature")
plt.plot(x,X@model_w+model_b,label="Predicted Value");plt.xlabel("x");plt.ylabel("y");plt.legend()
plt.show()

#Feature Engineering With Complex Functions
x=np.arange(0,20,1)
y=np.cos(x/2)
X=np.c_[x,x**2,x**3,x**4,x**5,x**6,x**7,x**8,x**9,x**10,x**11,x**12,x**13]
X=zscore_normalize_features(X)
model_w,model_b=run_gradient_descent_feng(X,y,iterations=1000000,alpha = 1e-1)
plt.scatter(x,y,marker='x',c='r',label="Actual Value");plt.title("Normalized x,x**2,x**3 Feature")
plt.plot(x,X@model_w+model_b,label="Predicted Value");plt.xlabel("x");plt.ylabel("y");plt.legend();
plt.show()