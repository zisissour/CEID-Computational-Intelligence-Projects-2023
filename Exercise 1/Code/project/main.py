import pandas as pd
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import keras

#Reading the dataset

data_raw  = pd.read_csv('./dataset.csv', delimiter=';', low_memory=False,encoding="utf-8-sig")

#Get rid of the user's name
#data_raw.pop('user')

data_processed = data_raw




###################################### Data preprocessing ##################################################



##Turning categorical values into numerical ones##

#Transforimng categorical features using ordinal encoder
oe = OrdinalEncoder()
oe.fit(data_processed[["user","gender"]])
data_processed[["user","gender"]] = oe.transform(data_processed[["user","gender"]]) 

#Transforing labels using label encoder
le = LabelEncoder()
data_processed.Class = le.fit_transform(data_processed.Class)



#Split features and labels
features = data_processed.copy()
labels = features.pop('Class')

#Replace commas with dots to recognize floats
features = features.replace(',','.', regex=True)


###############################Centering###################################################

scaler = StandardScaler(with_mean=True, with_std=False)
features = scaler.fit_transform(features)


##########################Normalize (MinMaxScaling)##################################


#Make the scaler
scaler = MinMaxScaler()

#Scaling
features = scaler.fit_transform(features)

#Turning dataframes into numpy arrays
features = np.array(features)
labels = np.array(labels)



######################### Building the model #######################################

#Creating gradient descent optimizer
sgd = SGD(learning_rate=0.001, momentum=0.6)

##Making K-folds##
skf = StratifiedKFold(n_splits=5, shuffle=True)
skf.get_n_splits(features, labels)

fold = 1

accuracy = []
val_accuracy = []
loss = []
val_loss = []
mse = []
val_mse = []

mean_accuracy = []
mean_ce =[]
mean_mse = []

#Itterating over the folds
for train_index , test_index in skf.split(features,labels):
    
    #Getting features and labels for this fold
    training_features = features[train_index]
    training_labels = labels[train_index]
    testing_features = features[test_index]
    testing_labels = labels[test_index]

    #Creating model
    model = Sequential()

    #Creating hidden and output layers
    hidden_layer = Dense(12, activation='relu')
    output_layer = Dense(5, activation='softmax')

    #Adding layers to model and compiling
    model.add(hidden_layer)
    model.add(output_layer)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['mse','acc'])
    
    #Adding early stopping criterion
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, min_delta=0.001, restore_best_weights=True)
    
    filepath = './f_' + str(fold) + 'model'
    save_callback = keras.callbacks.ModelCheckpoint(filepath=filepath, save_weights_only=False, monitor='val_loss', mode='min', save_best_only=True)


    #Fitting model with the data
    history = model.fit(x=training_features, y=training_labels, epochs=600, batch_size=128,
              validation_data=(testing_features, testing_labels), callbacks=[callback,save_callback],
              verbose=2, use_multiprocessing=True)

    #Getting the evaluation of our model
    eval = model.evaluate(x=testing_features, y=testing_labels, verbose=0, use_multiprocessing=True)

    #Print results
    print('Fold ' + str(fold) + '  loss: \n', eval[0])
    print('Fold ' + str(fold) + '  mse: \n', eval[1])
    print('Fold ' + str(fold) + '  accuracy: \n', eval[2])
    

    ####Saving results######
    mean_accuracy.append(eval[2])
    mean_ce.append(eval[0])
    mean_mse.append(eval[1])
    
     
    accuracy.append(history.history['acc'])
    val_accuracy.append(history.history['val_acc'])
    loss.append(history.history['loss'])
    val_loss.append(history.history['val_loss'])
    mse.append(history.history['mse'])
    val_mse.append(history.history['val_mse'])


    fold +=1

##########Calculating mean values##############

mean_accuracy = np.mean(mean_accuracy)
mean_ce = np.mean(mean_ce)
mean_mse = np.mean(mean_mse)

print('Mean accuracy: ', mean_accuracy)
print('Mean CE: ', mean_ce)
print('Mean mse: ', mean_mse)

##########################Plotting###########################

for i in range(0,5):
    plt.plot(accuracy[i])
    plt.plot(val_accuracy[i])
    plt.title('Fold ' + str(i+1)+ " Accuracy per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('fold_'+str(i+1)+'_acc.jpg', bbox_inches='tight', dpi=250)
    plt.close()

    plt.plot(loss[i])
    plt.plot(val_loss[i])
    plt.title('Fold ' +str(i+1)+ " CE per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy")
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('fold_'+str(i+1)+'_ce.jpg', bbox_inches='tight', dpi=250)
    plt.close()

    plt.plot(mse[i])
    plt.plot(val_mse[i])
    plt.title('Fold ' +str(i+1)+ " MSE per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('fold_'+str(i+1)+'_mse.jpg', bbox_inches='tight', dpi=250)
    plt.close()






