import pandas as pd
import keras
import numpy as np
import random
#from sklearn.model_selection import KFold , StratifiedKFold

#Reading the dataset

data_raw  = pd.read_csv('./dataset.csv', delimiter=';', low_memory=False)

#Data preprocessing

#Turning categorical values into numerical ones

#Itterate over all the columns containing categorical values

for column in data_raw['user','gender','class']:
    columnObj = data_raw[column]

    layer = keras.layers.StringLookup() #Creating string covnersion layer
    layer.adapt(columnObj.values) #Adapting layer to column data (making the vocabulary)
    x = layer(columnObj.values) #Getting data through the layer

    data_raw[column] = np.array(x) #Replacing values int the original dataset


print('Final: ')
print(data_raw)