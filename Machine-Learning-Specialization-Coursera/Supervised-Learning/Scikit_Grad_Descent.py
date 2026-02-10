#Implement Linear Regression Using Scikit-Learn 
#Importing Modules
import numpy as np,matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor #Gradient Descent Model;Performs Best With Normalized Inputs
from sklearn.preprocessing import StandardScaler #Performs Z-Score Normalization
#Refer To As Standard Score
from lab_utils_multi import  load_house_data
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

#Loading The Data Set
X_train,y_train=load_house_data()
X_feat=['Size (sqft)','Bedrooms','Floors','Age']

print()
#Normalizing The Training Data
scaler=StandardScaler()
normx=scaler.fit_transform(X_train)
print(f"Peak To Peak Range By column In Raw X:{np.ptp(X_train,axis=0)}")   
print(f"Peak To Peak Range By column In Normalized X:{np.ptp(normx,axis=0)}")

print()
#Creating And Fitting The Regression Model
sgdr=SGDRegressor(max_iter=1000)
sgdr.fit(normx,y_train)
print(sgdr)
print(f"Number Of Iterations Completed:{sgdr.n_iter_}, Number Of Weight Updates:{sgdr.t_}")

print()
#Viewing Parameters
norm_b=sgdr.intercept_
norm_w=sgdr.coef_
print(f"Model Parameters-> w:{norm_w},b:{norm_b}")

print()
#Making Predictions 
'''
Predicting The Targets Of The Training Data
'''
#Using sgdr.predict()
pred_sgd=sgdr.predict(normx)
#Using w And b
y_pred=np.dot(normx,norm_w)+norm_b
print(f"Prediction Using np.dot() And sgdr.predict Match:{(y_pred==pred_sgd).all()}")
print(f"Prediction On Training Set:\n{y_pred[:4]}" )
print(f"Target Values \n{y_train[:4]}")

print()
#Plotting The Results
'''
Plotting Predictions And Targets Vs Original Features 
'''
fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train,label='Target')
    ax[i].set_xlabel(X_feat[i])
    ax[i].scatter(X_train[:,i],y_pred,color=dlc["dlorange"], label='predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("Target Vs Prediction Using Z-Score Normalized Model")
plt.show()
print()