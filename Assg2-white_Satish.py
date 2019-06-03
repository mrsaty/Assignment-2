#!/usr/bin/env python
# coding: utf-8

# In[404]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[405]:


data = pd.read_csv('/home/satish/Downloads/winequality-white.csv', sep = ';')
data[:10]   #to read the data


# In[406]:


data.describe()


# In[407]:


#x = data[data.columns[0:4]] #displays first, second, third columns
#x


# In[408]:


data = (data - data.mean()) / data.std()
data[:10]


# In[409]:


_m = data.shape[0]
k = .8*_m
n = data.shape[1]
print(_m, n, sep = ",")
import math
print(math.ceil(k))


# In[410]:


x = np.asarray(data.iloc[:math.ceil(k),:n-1])
y = np.asarray(data.iloc[:math.ceil(k),n-1:])
m = len(y)
ones = np.ones([m,1])
x = np.concatenate([ones,x],1)
theta = np.zeros([1,n])               #theta is a row matrix of size n

#Cost Function
'''
cost function is the mean squared error difference between
the predicted value and the actual value
'''
def costfn(x, y, theta):
    m = len(y)
    _cost = 1/(2*m)*(np.sum(np.power((x.dot(theta.T) - y),2)))
    return _cost

#Gradient Descent Algorithm
'''
This algorithm is used to determine the minimum (optimum) value
of cost function and the optimized value of theta
It calculates derivative of cost function wrt. different theta's
and simultaneously updates all the values of theta 
'''
def gradDes(x,y,theta,alpha,iters):
    cost = np.zeros(iters) #a row matrix of size iters or an array of size iters
    m = len(y)
    for i in range(iters):
        theta = theta - (alpha/m)*(np.sum((x*(x.dot(theta.T) - y)), axis = 0))
        cost[i] = costfn(x,y,theta)
    return theta, cost


alpha = 0.1
iters = 100
_theta, cost = gradDes(x,y,theta,alpha,iters)
print("finalTheta = ",_theta)
print()

finalCost = costfn(x,y,_theta)
print("finalCost = ",finalCost)


# In[411]:


#cost vs iterations plot, at alpha = 0.1

plt.plot(np.arange(iters),cost)
plt.ylabel('Cost Function')
plt.xlabel('No. of Iterations')
plt.title('cost vs iterations') 


# In[412]:


'''
effects of having different learning rates
'''

#at alpha > 0.1, the graph does not converge
#at alpha = 0.1, graph converges at around 60th iterations and finalCost =  0.37253211642296413
#at alpha = 0.01, graph converges at around 400th iterations and finalCost =  0.37253474715550733
#at alpha = 0.001, graph converges at around 4000th iterations and finalCost =  0.372535019554979
#at alpha = 0.0001, graph converges at around 40000th iterations and finalCost =  0.3725350468900399

#from the above data of alpha's, we can say that alpha(optimum) = 0.1


# In[413]:


#y_pred = np.concatenate([np.ones([_m-3920,1]),(np.asarray(data.iloc[3920:,:n-1]))],1).dot(_theta.T)
#y_pred

#y_test = np.asarray(data.iloc[3920:,n-1:])
#y_test.shape

#from sklearn.metrics import r2_score
#r2 = r2_score(y_test, y_pred)
#print(r2)     #0.15128756455886772

#from sklearn.metrics import mean_squared_error as mse
#_mse = mse(y_test, y_pred)
#print(_mse)   #0.6505409800535203


# In[ ]:




