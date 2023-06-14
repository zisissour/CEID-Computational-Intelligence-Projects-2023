import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random

#Load model
model  = keras.models.load_model('./model')


for i in range(10):
    #Load solution
    solution_path = "Experiments/exp_"+str(i+1)+"/exp_"+str(i+1)+"_solution.txt"
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

    #Save results
    
    file = open("Experiments/exp_"+str(i+1)+"/exp_"+str(i+1)+"_evaluation.txt","w")
    file.write("------Prediction Probabilities------\nsitting: " + str(results[0][0]) +"\nsittingdown: "+str(results[0][1])+"\nstanding: " + str(results[0][2]) +"\nstandingup: " + str(results[0][3]) +"\nwalking: " + str(results[0][4]))
    file.close()