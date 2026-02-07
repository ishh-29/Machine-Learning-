#Extend The Regression Model Routines To Support Multiple Features

import copy,math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
#Reduced Display Precision 
np.set_printoptions(precision=2)

#Loading Data Sets
X_train=np.array([[2104,5,1,45],
                  [1416,3,2,40],
                  [852,2,1,35]])
y_train=np.array([460,232,178])
#Data Stored In Numpy Array/Matrix
print(f"X-Shape: {X_train.shape}, X Type: {type(X_train)})")
print(X_train)
print(f"y-Shape: {y_train.shape}, y Type: {type(y_train)}")
print(y_train)

#Parameter Vectors w,b
b_init=785.1811367994083
w_init=np.array([0.39133535,18.75376741,
                 -53.36032453,-26.42131618])
print(f"w_init Shape: {w_init.shape},b_init: {type(b_init)}")

#Single Prediction
def predict(x,w,b):
    p=np.dot(x,w)+b
    return p

#Getting A Row From Our Training Data
vec_x=X_train[0,:]
print(f"vec_x-Shape: {vec_x.shape},x_vec Value: {vec_x}")
#Making A Prediction
f_wb=predict(vec_x,w_init,b_init)
print(f"f_wb-Shape: {f_wb.shape},Prediction: {f_wb}")

#Computing Cost With Multiple Variables
def compute_cost(X,y,w,b):
    m=X.shape[0]
    cost=0.0
    for i in range(m):
        f_wb_i=np.dot(X[i],w)+b
        cost+=(f_wb_i-y[i])**2
    cost/=(2*m)
    return cost

cost=compute_cost(X_train,y_train,w_init,b_init)
print(f"Cost at w_init,b_init: {cost}")

#Gradient Descent With Multiple Variables
def compute_gradient(X,y,w,b):
    m,n=X.shape
    dj_dw=np.zeros((n,))
    dj_db=0
    for i in range(m):
        err=(np.dot(X[i],w)+b)-y[i]
        dj_db+=err
        for j in range(n):
            dj_dw+=err*X[i,j]
    dj_dw/=m
    dj_db/=m
    return dj_db,dj_dw
#Computing And Displaying Gradient 
tmp_dj_db,tmp_dj_dw=compute_gradient(X_train,y_train,w_init,b_init)
print(f'dj_db at initial w,b:{tmp_dj_db}')
print(f'dj_dw at initial w,b:\n {tmp_dj_dw}')
#Gradient Descent With Multiple Variables
def gradient_descent(X,y,w_in,b_in,cost,grad,alpha,num_iters):
    '''
    Performs Batch Gradient Descent To Learn Theta.Updates Theta By Taking num_iters 
    Gradient Steps With Learning Rate Aplha.
    Args:
    X (ndarray (m,n))   : Data, m Examples With n Features
    y (ndarray (m,))    : Target Values
    w_in (ndarray (n,)) : Initial Model Parameters  
    b_in (scalar)       : Initial Model Parameter
    cost                : Function To Compute Cost
    grad                : Function To Compute The Gradient
    alpha (float)       : Learning Rate
    num_iters (int)     : Number Of Iterations To Run Gradient Descent
    '''
    #To Store For Graphing Later
    j_arr=[]
    w=copy.deepcopy(w_in) #To Avoid Modifying Global w Within Function
    b=b_in 
    for i in range(num_iters):
        #Calculating The Gradient And Updating The Parameters
        dj_db,dj_dw=grad(X,y,w,b)   #None
        #Updating Parameters Using w,b,alpha And gradient
        w-=alpha*dj_dw       #None
        b-=alpha*dj_db       #None
        #Saving Cost J At Each Iteration
        if i<100000:      #Preventing Resource Exhaustion 
            j_arr.append(cost(X,y,w,b))
        #Printing Cost Every At Intervals 10 Times
        if i%math.ceil(num_iters/10)==0:
            print(f"Iteration {i:4d}: Cost {j_arr[-1]:8.2f}")
    return w,b,j_arr

#Initalizing Parameters
w_prime=np.zeros_like(w_init)
b_prime=0.
#Some Gradient Descent Settings
iters=1000
alpha=5.0e-7
#Running Gradient Descent 
w_final,b_final,j_arr=gradient_descent(X_train,y_train,w_prime,b_prime,
                                        compute_cost,compute_gradient, 
                                        alpha,iters)
print(f"b,w Found By Gradient Descent: {b_final:0.2f},{w_final}")
m,_ =X_train.shape
for i in range(m):
    print(f"Prediction:{np.dot(X_train[i],w_final)+b_final:0.2f},Target Value:{y_train[i]}")
#Plotting Cost Vs Iteration  
fig,(ax1,ax2)=plt.subplots(1,2,constrained_layout=True,figsize=(12,4))
ax1.plot(j_arr)
ax2.plot(100+np.arange(len(j_arr[100:])),j_arr[100:])
ax1.set_title("Cost vs. iteration");ax2.set_title("Cost Vs.Iteration(Tail)")
ax1.set_ylabel('Cost')             ;ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;ax2.set_xlabel('Iteration Step') 
plt.show()