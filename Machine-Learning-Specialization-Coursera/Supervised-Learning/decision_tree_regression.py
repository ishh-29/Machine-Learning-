#Implementation Of Decision Tree Regression
#Importing Required Modules And Libraries
import numpy as np, matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor,export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Creating A Sample Dataset
np.random.seed(42)
x=np.sort(5*np.random.rand(100,1),axis=0)
y=np.sin(x).ravel()+np.random.normal(0,0.1,x.shape[0])

#Plotting The Dataset
plt.scatter(x,y,color='red',label='Data')
plt.title("Synthetic Dataset")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()

#Splitting The Dataset
#Training->70% | Testing ->30%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

#Initializing The Decision Tree Regressor
t_regressor=DecisionTreeRegressor(max_depth=4,random_state=42)

#Fitting The Model
t_regressor.fit(x_train,y_train)

#Predicting A New Value
pred=t_regressor.predict(x_test)
err=mean_squared_error(y_test,pred)
print(f"Mean Squared Error:{err:.4f}")

#Visualization
x_ax= np.arange(min(x),max(x),0.01)[:,np.newaxis]
pred_ax=t_regressor.predict(x_ax)
plt.figure(figsize=(10,6))
plt.scatter(x,y,color='red',label='Data')
plt.plot(x_ax,pred_ax,color='blue',label='Model Prediction')
plt.title("Decision Tree Regression")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()

#Visualizing The Tree Structure
from sklearn.tree import plot_tree
plt.figure(figsize=(20,10))
plot_tree(
    t_regressor,
    feature_names=["Feature"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Structure")
plt.show()
