#Implementing KNN Algorithm 
#Importing Required Modules And Libraries
import numpy as np
from collections import Counter

#Euclidean Distance Function
def euclid(p1,p2):
    return np.sqrt(np.sum((np.array(p1)-np.array(p2))**2))

#KNN Prediction Function
def knn(train_data,train_label,testpts,k):
    dist=[]
    for i in range(len(train_data)):
        d=euclid(testpts,train_data[i])
        dist.append((d,train_label[i]))
    dist.sort(key=lambda x:x[0])
    near_label=[i for i,j in dist[:k]]
    return Counter(near_label).most_common(1)[0][0]

#Training Data,Labels And Test Points
train_data=[[1,2],[2,3],[3,4],[6,7],[7,8]]
train_label=['A','A','A','B','B']
testpts=[4,5]
k=3

#Prediction
pred=knn(train_data,train_label,testpts,k)
print(pred)