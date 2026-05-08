#Implementation Of Random Forest Regression 
#Importing Required Modules And Libraries
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.metrics import r2_score as r2,mean_squared_error as MSE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree 

#Loading The Datasets
df=pd.read_csv('Position_Salaries.csv')
print(df)
df.info()

#Extracting Features And Target Variable
x=df.iloc[:,1:2].values
y=df.iloc[:,2].values

#Encoding Categorical Columns
lbl_enc=LabelEncoder()
for i in df.select_dtypes(include=['object']).columns:
    df[i]=lbl_enc.fit_transform(df[i])

#Splitting The Dataset
x_train,x_test,y_train,y_test=ttsplit(x,y,test_size=0.2,random_state=42)

#Initializing The Random Forest Model 
rf_regress=RandomForestRegressor(n_estimators=100,random_state=42,oob_score=True) #Out Of Bag Score
rf_regress.fit(x_train,y_train)

#Making Predictions
print("Out-Of-Bag Score:",rf_regress.oob_score_)
pred=rf_regress.predict(x_test)
print("Mean Squared Error",MSE(y_test,pred))
print("R-2 Score:",r2(y_test,pred))

#Visualizing The Results
xgrid=np.arange(min(x),max(x),0.01).reshape(-1,1)
plt.scatter(x,y,color='blue',label="Actual Data")
plt.plot(xgrid,rf_regress.predict(xgrid),color='green',label="Random Forest Prediction")
plt.title("Random Forest Regression Results")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

#Visualizing A Single Decision Tree
tree_plot=rf_regress.estimators_[0]
plt.figure(figsize=(20,10))
plot_tree(tree_plot,feature_names=df.columns.tolist(),filled=True,rounded=True,fontsize=10)
plt.title("Decision Tree From Random Forest")
plt.show()