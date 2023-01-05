#!pip install tensorflow
#!pip install numpy 
#!pip install pandas 
#!pip install -U scikit-learn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models

model=models.Sequential()
model.add(layers.Dense(64,activation='relu',input_dim=(data.drop(columns=['Y'])).shape[1]))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(128*2,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='sgd',metrics='accuracy')

X=np.array(data.drop(columns=['Y']))
Y=np.array(data['Y']).reshape(-1,1)
model.fit(X,Y,epochs=300)

print(model.predict([[1,0,0]]))