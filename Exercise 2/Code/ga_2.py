#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from deap import base,creator, tools


# In[2]:


dataset = pd.read_csv('dataset.csv', delimiter=';', low_memory='False')


# In[3]:


#Drop columns that are not needed
dataset = dataset.drop(['user','gender','age','how_tall_in_meters','weight','body_mass_index'], axis=1)


# In[4]:


#Centering
scaler = StandardScaler(with_mean=True, with_std=False)
dataset[['x1','y1','z1','x2','y2','z2','x3','y3','z3','x4','y4','z4']] = scaler.fit_transform(dataset[['x1','y1','z1','x2','y2','z2','x3','y3','z3','x4','y4','z4']])

#MinMax Scaling
scaler = MinMaxScaler();
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

w = []
for _ in range(0,12):
    w.append(1.0)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


toolbox = base.Toolbox()
toolbox.register("attribute", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=12)
                 
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(v):
    c=0.5
    
    v= np.array(v).reshape(1,-1)
    other_states_sum = 0
    
    for state in states_means:
        other_states_sum += cosine_similarity(v,state.reshape(1,-1))
    
    f = (cosine_similarity(v,sitting_mean.reshape(1,-1)) + c*(1 - 0.25 * other_states_sum)) / (1 + c)
    

    
    return f[0][0]


toolbox.register("mate", tools.cxSimulatedBinary, eta=10)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


    


# In[10]:


def ga():
    pop = toolbox.population(n=100)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 1000

    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for g in range(NGEN):
        
        print("\n----GEN "+ str(g+1) +"----\n")
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring =list(map(toolbox.clone, offspring))
        
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1,child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
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

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)


    


# In[11]:


ga()

