import pandas as pd
import keras
import numpy as np
import random
#from sklearn.model_selection import KFold , StratifiedKFold

#Reading the dataset

data_raw  = pd.read_csv('./dataset.csv', delimiter=';', low_memory=False)

#Data preprocessing

#Turning categorical values into numerical values

categorical_data = data_raw['user','gender','class'].copy()

for 

layer = keras.layers.StringLookup()
layer.adapt(data)

x = layer(data)
print(np.array(x))

data_raw['class'] = np.array(x)
