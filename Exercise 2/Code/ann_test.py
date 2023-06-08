import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random

#Load model
model  = keras.models.load_model('./model')

#Load solution
solution_path = "Experiments/exp_1/exp_1_solution.txt"
ga = np.genfromtxt(solution_path,delimiter=',',skip_header=0)

#Apply minmax scaling
scaler = MinMaxScaler()
ga = scaler.fit_transform(ga.reshape(-1,1))
ga = ga.reshape(1,-1)

#Padding with zeros for the missing inputs
data = [0,0,0,0,0,0]
data.extend(ga[0])

#Get predictions
results = model.predict([data])

print("------Prediction Probabilities------\nsitting: " + str(results[0][0]) +"\nsittingdown: "+str(results[0][1])+"\nstanding: " + str(results[0][2]) +"\nstandingup: " + str(results[0][3]) +"\nwalking: " + str(results[0][4]))