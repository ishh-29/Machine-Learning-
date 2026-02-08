'''
Feature Scaling:

Utilize The Multiple Variables Routines
Run Gradient Descent On A Data Set With Multiple Features
Improve Performance Of Gradient Descent By Feature Scaling Using Z-Score Normalization
'''
import numpy as np,matplotlib.pyplot as plt
from lab_utils_multi import  load_house_data,run_gradient_descent 
from lab_utils_multi import  norm_plot,plt_equal_scale,plot_cost_i_w
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

#Loading The Data Sets
X_train,y_train=load_house_data()
X_feat=['Size (sqft)','Bedrooms','Floors','Age']

#Plotting Feature Vs Price

fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train)
    ax[i].set_xlabel(X_feat[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()

#Setting Alpha=9.9e-7
_,_,hist=run_gradient_descent(X_train,y_train,10,alpha=9.9e-7)

#Plotting The Errorous Gradient Descent 
plot_cost_i_w(X_train,y_train,hist)

#Setting Alpha=9e-7
_,_,hist=run_gradient_descent(X_train,y_train,10,alpha=9e-7)

#Plotting The Errorous Gradient Descent 
plot_cost_i_w(X_train,y_train,hist)

#Setting Alpha=1e-7
_,_,hist=run_gradient_descent(X_train, y_train,10,alpha=1e-7)

#Plotting The Correct Gradient Descent
plot_cost_i_w(X_train,y_train,hist)

#Z-Score Normalization 
def z_score(X):
    """
    Computes X, Z-Score Normalized By Column
    Args:
      X (ndarray (m,n))     :Input Data,M Examples,N Features  
    Returns:
      X_norm (ndarray (m,n)):Input Normalized By Column
      mu (ndarray (n,))     :Mean Of Each Feature
      sigma (ndarray (n,))  :Standard Deviation Of Each Feature
    """
    #Finding The Mean Of Each Feature/Column 
    mu=np.mean(X,axis=0) #mu With Shape (n,)
    #Finding The Standard Deviation Of Each Column/Feature 
    sigma=np.std(X,axis=0) #Sigma With Shape (n,)
    #Element-wise Subtraction Of mu For That Column From Each Example
    #Divide By std For That Column
    normx=(X-mu)/sigma
    return (normx,mu,sigma)

#Checking The Results
from sklearn.preprocessing import scale
scale(X_train,axis=0,with_mean=True,with_std=True,copy=True)

mu=np.mean(X_train,axis=0)
sigma=np.std(X_train,axis=0)
meanx=(X_train-mu)
normx=(X_train-mu)/sigma

#Plotting
fig,ax=plt.subplots(1,3,figsize=(12,3))
ax[0].scatter(X_train[:,0],X_train[:,3])
ax[0].set_xlabel(X_feat[0]);ax[0].set_ylabel(X_feat[3]);
ax[0].set_title("Unnormalized")
ax[0].axis('equal')

ax[1].scatter(meanx[:,0],meanx[:,3])
ax[1].set_xlabel(X_feat[0]); ax[0].set_ylabel(X_feat[3]);
ax[1].set_title(r"X-$\mu$")
ax[1].axis('equal')

ax[2].scatter(normx[:,0],normx[:,3])
ax[2].set_xlabel(X_feat[0]); ax[0].set_ylabel(X_feat[3]);
ax[2].set_title(r"Z-Score Normalized")
ax[2].axis('equal')
plt.tight_layout(rect=[0,0.03,1,0.95])
fig.suptitle("Distribution Of Features Before,During,After Normalization")
plt.show()

#Normalizing The Original Features
normx,muX,sigmaX=z_score(X_train)
print(f"X_mu={muX},\nX_sigma={sigmaX}")
print(f"Peak to Peak range by column in Raw X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(normx,axis=0)}")

#Plotting The Results
#Before Normalization
fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    norm_plot(ax[i],X_train[:,i],)
    ax[i].set_xlabel(X_feat[i])
ax[0].set_ylabel("Count");
fig.suptitle("Distribution Of Features Before Normalization")
plt.show()
#After Normalization
fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    norm_plot(ax[i],normx[:,i],)
    ax[i].set_xlabel(X_feat[i])
ax[0].set_ylabel("count"); 
fig.suptitle("Distribution Of Features After nNrmalization")
plt.show()

#Again Running Gradient Descent Algorithm
norm_w,norm_b,hist=run_gradient_descent(normx,y_train,1000,1.0e-1)
#Plotting The Original Features With Normalized Results
#predict target using normalized features
m=normx.shape[0]
yp=np.zeros(m)
for i in range(m):
    yp[i]=np.dot(normx[i],norm_w)+norm_b  
fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train,label='target')
    ax[i].set_xlabel(X_feat[i])
    ax[i].scatter(X_train[:,i],yp,color=dlc["dlorange"],label='predict')
ax[0].set_ylabel("Price");ax[0].legend();
fig.suptitle("Target Vs Prediction Using Z-Score Normalized Model")
plt.show()

'''
NOTE: Any Predictions Using The Parameters Learned From A Normalized Training
      Set Must Also Be Normalized. 
'''
x_house=np.array([1200,3,1,40])
house_norm=(x_house-muX)/sigmaX
print(house_norm)
house_predict=np.dot(house_norm,norm_w)+norm_b
print(f"Predicted Price Of A House With 1200 sqft,3 Bedrooms,1 Floor,40 Years Old=${house_predict*1000:0.0f}")
plt_equal_scale(X_train,normx,y_train)