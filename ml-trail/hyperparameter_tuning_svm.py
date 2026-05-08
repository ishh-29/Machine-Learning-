#SVM Hyperparameter Tuning Using GridSearchCV
#Importing Required Modules And Libraries
import numpy as np, pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC

#Loading Dataset
cancer=load_breast_cancer()
df_feature=pd.DataFrame(cancer['data'],
                        columns=cancer['feature_names'])
df_target=pd.DataFrame(cancer['target'],
                       columns=['Cancer'])
print("Feature Variables:")
print(df_feature.info())
print("Dataframe:",df_feature.head())

#Splitting The Data Into Training And Testing Sets
x_train,x_test,y_train,y_test=train_test_split(
    df_feature,
    np.ravel(df_target),
    test_size=0.30,
    random_state=101
)

#Initializing SVM Model
svm=SVC()
svm.fit(x_train,y_train)
pred=svm.predict(x_test)
print(classification_report(y_test,pred))

#Hyperparameter Tuning 
param={'C':[0.1,1,10,100,1000],
       'gamma':[1,0.1,0.01,0.001,0.0001],
       'kernel':['rbf']}
grid=GridSearchCV(SVC(),param,refit=True,verbose=3)
grid.fit(x_train,y_train)
print(grid.best_params_)
print(grid.best_estimator_)

#Evaluating Optimized Model
grid_pred=grid.predict(x_test)
print(classification_report(y_test,grid_pred))
