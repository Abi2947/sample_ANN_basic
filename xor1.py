#!pip install pandas
#!pip install numpy

import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt

def generate_dataset(dims):

    def xor_output(x):
        return x%2


    df=pd.DataFrame()
    for i in range(0,dims):
        while True: 
            A=[random.randrange(0,2) for i in range(0,2**dims)]
            if A.count(1)==(2**dims)/2:
                break
        df[str(i)]=A

    op=0
    for i in df.columns:
        op+=df[i]

    df['Y']=op
    df['Y']=df['Y'].apply(xor_output)
    df
    return df 

dims=int(input('Enter number of dims: '))
data=generate_dataset(dims)
data

def generate_weights(data):
    print(data.columns)
    n=len(data.columns)-1
    return [random.uniform(0,1) for i in range(n)],random.uniform(-1,1)


def getting_proper_data(data,weights,bias):
    features=np.array(data.drop(columns=['Y']))
    labels=np.array(data['Y']).reshape(-1,1)
    weights=np.array(weights)
    bias=np.array(bias)
    return features,labels,weights,bias

alpha=10**-3
weights,bias=generate_weights(data)
features,labels,weights,bias=getting_proper_data(data,weights,bias)
weights,bias=train_model_perceptron(weights,bias,alpha,features,labels,3500)

print(weights)
print(bias)

print('\nPredictor: ')
op=sigmoid(np.dot(weights,[1,1,1])+bias)
print(op)