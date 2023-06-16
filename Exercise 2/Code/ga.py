#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from deap import base, creator, tools

from matplotlib import pyplot as plt


# In[2]:


dataset = pd.read_csv('dataset.csv', delimiter=';', low_memory='False')


# In[3]:


#Drop columns that are not needed
dataset = dataset.drop(['user','gender','age','how_tall_in_meters','weight','body_mass_index'], axis=1)


# In[4]:


#Centering
scaler = StandardScaler(with_mean=True, with_std=False)
dataset[['x1','y1','z1','x2','y2','z2','x3','y3','z3','x4','y4','z4']] = scaler.fit_transform(dataset[['x1','y1','z1','x2','y2','z2','x3','y3','z3','x4','y4','z4']])


# In[5]:


#Getting the means
means = dataset.groupby('Class')[['x1','y1','z1','x2','y2','z2','x3','y3','z3','x4','y4','z4']].mean()


# In[6]:


#Separating means for the sitting state and the other states
sitting_mean = means.T.pop('sitting').T
states_means = means.T.drop('sitting', axis=1).T

sitting_mean


# In[7]:


states_means


# In[8]:


states_means = np.array(states_means)
sitting_mean = np.array(sitting_mean)


# In[9]:


#Making the genetic algorithm

#We want to maximize fitness thus weights=1
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

#Describing the individual as a list of 12 integers from -617 to 533
toolbox = base.Toolbox()
toolbox.register("attribute", random.randint, -617,533)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=12)                 
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#Defining the evaluation function

def evaluate(v):
    c=0.1
    
    v= np.array(v).reshape(1,-1)
    other_states_sum = 0
    
    for state in states_means:
        other_states_sum += cosine_similarity(v,state.reshape(1,-1))
    
    f = ((cosine_similarity(v,sitting_mean.reshape(1,-1)) + c*(1 - 0.25 * other_states_sum)))/ (1 + c) + 1
    
    return f[0]


#Registering genetic operators
toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=0.01, low=-617, up=533)
toolbox.register("mutate", tools.mutUniformInt, low=-617,up=533, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=40)
toolbox.register("evaluate", evaluate)


    


# In[10]:


#The actual algorithm

def ga():
    
    #Create initial population
    pop = toolbox.population(n=200)
    CXPB, NGEN = 0.6, 1000
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    
    
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    
    
    g=0
    
    max_fits=1
    prev_max_fits=1
    max_fits_list=[]
    
    bad_improvement_counter=0
    
    best_solution=[]
        
    while g<NGEN and bad_improvement_counter<20 and max_fits<2:
        
        
        print("\n----GEN "+ str(g+1) +"----\n")
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        
        # Clone the selected individuals
             
        offspring =list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            
            #Apply crossover according to crossover prob.
            if random.random() < CXPB:
                toolbox.mate(child1,child2)
                del child1.fitness.values
                del child2.fitness.values
        
        #Always apply mutation. Mutation prob. is given as argument in mutation function
        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values
        
        
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        #Get best solution
        best_solution = pop[fits.index(max(fits))]
        
        #Get best evaluation for current generation
        prev_max_fits = max_fits
        max_fits = max(fits)
        max_fits_list.append(max_fits)
        
        #Get improvemnet
        improvement = (max_fits/prev_max_fits)-1
        
        #If too small increase counter
        if improvement < 0.001:
            bad_improvement_counter+=1
        else:
            bad_improvement_counter=0
        
        
        #Printing statistics for each generation
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        
        g+=1
        


    return best_solution, max_fits_list,g
    


# In[11]:


#########Plotting and getting exp. results#########

solutions=[]
max_fits=[]
generations=[]

for _ in range(20):
    solution, max_fit,gen = ga()
    solutions.append(solution)
    max_fits.append(max_fit)
    generations.append(gen)
    
mean_fit = pd.DataFrame(max_fits).mean()


# In[12]:


experiment_name = "expe"
experiment_path = "Experiments/"+experiment_name+"/"

if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)

plt.title("Max evaluation per generation")
plt.xlabel("Generation")
plt.ylabel("Max Evaluation")
plt.plot(mean_fit)
plt.savefig(experiment_path+experiment_name+".jpg")


# In[13]:


f = open(experiment_path+experiment_name+"_gens.txt","w")
f.write("Mean number of generations: " +str(np.mean(generations)) +"\nMean max evaluation: "+str(max(mean_fit)))
f.close()


# In[14]:


best_solution = solutions[max_fits.index(max(max_fits))]
np.savetxt(experiment_path+experiment_name+"_solution.txt", best_solution, delimiter=',')

