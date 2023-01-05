#!pip install tensorflow
#!pip install numpy 
#!pip install pandas 
#!pip install -U scikit-learn

import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('flower_dataset.csv')
df
features=np.array(df[['X1','X2']])
labels=np.array(df['Y']).reshape(-1,1)

def sigmoid(x):    
    return 1/(1+np.exp(-x))

def cost(P,A):
    return (P-A)**2

def d_cost(P,A):
    h=0.001
    return (cost(P+h,A)-cost(P,A))/h

def d_sigmoid(x):
    h=0.0001
    return (sigmoid(x+h)-sigmoid(x))/h




def train_model_perceptron(weights,bias,alpha,features,labels,epochs):
    for epoch in range(int(epochs)): 
        z=np.dot(features,weights)+bias
        P=[sigmoid(i) for i in z]
        for index in range(len(z)):
            weights=[weights[indx]-alpha*d_cost(P[index],labels[index][0])*d_sigmoid(z[index])*features[index][indx] for indx in range(len(weights))]    
            bias=bias-alpha*d_cost(P[index],labels[index][0])*d_sigmoid(z[index])
    return weights,bias

weights=np.array([0.25,0.65])
bias=0.25
alpha=0.001

weights,bias=train_model_perceptron(weights,bias,alpha,features,labels,3500)
input_val=[float(i) for i in (input('Enter value of length and width seperate with comma: ')).split(',')]
print(input_val)
z=np.dot(input_val,weights)+bias
output=sigmoid(z)
if output>=0.50:
    output=1 
else: 
    output=0
print(output)


plt.scatter(x=df['X1'],y=df['X2'])
plt.scatter(x=input_val[0],y=input_val[1],color='red')

plt.show()