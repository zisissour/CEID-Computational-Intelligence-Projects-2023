import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import keras

#Reading the dataset

data_raw  = pd.read_csv('./dataset.csv', delimiter=';', low_memory=False,encoding="utf-8-sig")

#Get rid of the user's name
data_raw.pop('user')

data_processed = data_raw




######## Data preprocessing ##########



##Turning categorical values into numerical ones##


#Itterate over all the columns containing categorical values
for column in data_processed[['gender','class']]:
    columnObj = data_processed[column]

    layer = keras.layers.StringLookup() #Creating string covnersion layer
    layer.adapt(columnObj.values) #Adapting layer to column data (making the vocabulary)
    x = layer(columnObj.values) #Getting data through the layer

    data_processed[column] = np.array(x) #Replacing values in the original dataset


#Split features and labels
features = data_processed.copy()
labels = features.pop('class')


##Normalize (MinMaxScaling)##


#Make the scaler
scaler = MinMaxScaler()

#Replace commas with dots to recognizes floats
features = features.replace(',','.', regex=True)

#Scaling
features = scaler.fit_transform(features)

features = np.array(features)
labels = np.array(labels)

##### Building the model #######

#Get learning rate, momentum, loss function and number of hidden units
lr = float(input("Enter learning rate:"))
m = float(input("Enter momentum: "))
loss_func = input("Enter loss function:")
hidden = int(input("Enter number of hidden units:"))

#Creating model
model = Sequential()

#Creating hidden and output layers
hidden_layer = Dense(hidden, input_shape=(17,), activation='relu')
output_layer = Dense(5, activation='softmax')

#Creating gradient descent optimizer with learning rate=0.001 and momentum=0
sgd = SGD(learning_rate=lr, momentum=m)

#Adding layers to model and compiling
model.add(hidden_layer)
model.add(output_layer)
model.compile(loss=loss_func, optimizer=sgd, metrics='acc')

##Making K-folds##


skf = StratifiedKFold(n_splits=5, shuffle=True)
skf.get_n_splits(features, labels)

fold = 1

for train_index , test_index in skf.split(features,labels):
    
    training_features = features[train_index]
    training_labels = tf.one_hot(labels[train_index],5)
    testing_features = features[test_index]
    testing_labels = tf.one_hot(labels[test_index],5)
    

    model.fit(x=training_features, y=training_labels, epochs=5, 
              validation_data=(testing_features, testing_labels),
              verbose=2, use_multiprocessing=True)

    eval = model.evaluate(testing_features, testing_labels, verbose=2, use_multiprocessing=True)

    f = open('Model Evaluations.txt', 'w')
    f.write('Learning Rate= ' +str(lr) +'Momentum= ' +str(m) +'Loss Function= ' +loss_func +'Hidden Units= '+str(hidden)+'\n')
    f.write('Fold='+ str(fold) +':\n'+str(eval))
    f.close()

    fold +=1




