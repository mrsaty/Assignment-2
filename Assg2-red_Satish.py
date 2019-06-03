#!/usr/bin/env python
# coding: utf-8

# In[273]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt


# In[274]:


data = pd.read_csv('/home/satish/Downloads/winequality-red.csv', sep = ';')
data[:10]


# In[275]:


data = (data - data.mean())/data.std()
data[:10]


# In[276]:


_m = data.shape[0]
k = .8*_m
n = data.shape[1]
print(_m, n, sep = ",")
import math
print(math.ceil(k))


# In[277]:


x = np.asarray(data.iloc[:math.ceil(k),:n-1])
y = np.asarray(data.iloc[:math.ceil(k),n-1:])
m = len(y)
ones = np.ones([m,1])
x = np.concatenate([ones,x],1)
theta = np.zeros([1,n])               #theta is a row matrix of size n

def costfn(x, y, theta):
    m = len(y)
    _cost = 1/(2*m)*(np.sum(np.power((x.dot(theta.T) - y),2)))
    return _cost

#cost = costfn(x,y,theta)
#print("cost = ",cost)

def gradDes(x,y,theta,alpha,iters):
    cost = np.zeros(iters) #a row matrix of size iters or an array of size iters
    m = len(y)
    for i in range(iters):
        theta = theta - (alpha/m)*(np.sum((x*(x.dot(theta.T) - y)), axis = 0))
        cost[i] = costfn(x,y,theta)
    return theta, cost


alpha = 0.1
iters = 70
_theta, cost = gradDes(x,y,theta,alpha,iters)
print("finalTheta = ",_theta)
print()

finalCost = costfn(x,y,_theta)
print("finalCost = ",finalCost)


# In[278]:


#cost vs iterations plot, at alpha = 0.1

plt.plot(np.arange(iters),cost)
plt.ylabel('Cost Function')
plt.xlabel('No. of Iterations')
plt.title('cost vs iterations')


# In[279]:


'''
effects of having different learning rates
'''

#at alpha > 0.1, the graph does not converge
#at alpha = 0.1, graph converges at around 35th iterations and finalCost =  0.31879075827752423
#at alpha = 0.01, graph converges at around 200th iterations and finalCost =  0.3187925519793649
#at alpha = 0.001, graph converges at around 2000th iterations and finalCost =  0.31879273410233244
#at alpha = 0.0001, graph converges at around 20000th iterations and finalCost =  0.3187927523423295


#from the above data of alpha's, we can say that alpha(optimum) = 0.1


# In[280]:


y_pred = np.concatenate([np.ones([_m-1280,1]),(np.asarray(data.iloc[1280:,:n-1]))],1).dot(_theta.T)
#y_pred
y_test = np.asarray(data.iloc[1280:,n-1:])
#y_test.shape
#from sklearn.metrics import r2_score
#r2 = r2_score(y_test, y_pred)
#print(r2)     #0.2805362993209275
#from sklearn.metrics import mean_squared_error as mse
#_mse = mse(y_test, y_pred)
#print(_mse)   #0.6626645626672407


# In[281]:


#plt.plot(y_pred[:20],y_test[:20])
#plt.xlabel('y_predicted')
#plt.ylabel('y_actual')


# In[282]:


#another way to plot a graph

#fig, ax = plt.subplots()  
#ax.plot(np.arange(iters), cost, 'r')  
#ax.set_xlabel('Iterations')  
#ax.set_ylabel('Cost')


# In[ ]:




