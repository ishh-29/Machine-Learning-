#Implementation Of The Model f_{w,b} For Linear Regression With One Variable

import numpy as np
import matplotlib.pyplot as plt

def model_output(x,w,b):
  m=x.shape[0]
  f_wb=np.zeros(m)
  for i in range(m):
    f_wb[i]=w*x[i]+b
  return f_wb

#Input Variables
x_train=np.array([1.0,2.0])
y_train=np.array([300.0,500.0])
print(f"x_train={x_train}")
print(f"y_train={y_train}")
#Number Of Training Examples m
print(f"x_train.shape:{x_train.shape}")
m=x_train.shape[0]
print(f"Number Of Training Examples m:{m}")
'''
Other Method Is len()
m=len(x_train)
print(f"Number Of Training Examples m:{m}")
'''
#Training Examples
i=0
x_i=x_train[i]
y_i=y_train[i]
print(f"(x^({i}),y^({i}))=(({x_i},{y_i}))")
#Plotting The Data
plt.scatter(x_train,y_train,marker="x",c="r")
plt.title("Housing Prices")
plt.ylabel("Prices (in 1000s of dollars)")
plt.xlabel("Size (1000 sqft)")
plt.show()

w=200
b=100
x_i=1.2
cost=w*x_i+b
print(f"${cost:.0f} Thousand Dollars")

temp_f=model_output(x_train,w,b,)
plt.plot(x_train,temp_f,c='b',label='Our Prediction')
plt.scatter(x_train,y_train,marker='x',c='r',label='Actual Values')
plt.title("Housing Prices")
plt.ylabel("Price (in 1000s of dollars)")
plt.xlabel("Size (1000 sqft)")
plt.legend()
plt.show()