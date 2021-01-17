# -*- coding: utf-8 -*-

import numpy as np
import random
import pandas as pd
import timeit
import pickle
from Crossover_Mutation import crossover, swift_mutation, update_stats
from Fundamental_Calculation_Functions import makespan
from NEH_Heuristic import NEH


car1 = pd.read_csv('car1.csv').values
car6 = pd.read_csv('car6.csv').values
reC05 = pd.read_csv('reC05.csv').values
reC07 = pd.read_csv('reC07.csv').values
reC19 = pd.read_csv('reC19.csv').values



def GA(dataset, M = 30, Pc = 1, Pminit = 0.8, theta = 0.99, D = 0.95): 
    """ 
    The main Genetic Algorithm function. It takes 
    as input the GA variants and a dataset
    where rows represent the jobs and columns 
    represent the machines. 
    It executes an iteration until maximun number of 
    solutions is reached, in which every time 2 parent 
    arrays are chosen and 2 childen are generated, 
    while then (randomly) only one of the children 
    replaces an old sequence of the population.
    
    GA variants:
        M: Population size
        Pc: Crossover probability
        Pminit: Initial mutation probability
        theta = 0.99
        D: Threshold parameter
        
    :modules used: numpy as np, random
    :returns: makespan (int), population (list)
    
    """       
    
    # Create initial popuplation and evaluate fitness (makespan) 
    fitness_values, population = NEH(dataset)  
    fitness_values = [fitness_values]   # Convert to list to handle faster
    population = [population.tolist()]  # Take NEH solution as a list of list
    for i in range(1, M):
        # Append random solution to population 
        population.append( 
                random.sample(range(len(population[0])), len(population[0])) )
        # Evaluate and append random-solutions' fitness value (makespan)
        fitness_values.append( 
                makespan( dataset[ np.array(population[i]) ] ) )
    
    
    # Convert into numpy arrays
    fitness_values = np.array(fitness_values)
    population = np.array(population)
    
    # Return sorted numpy arrays in descending order of fitness
    fitness_values, population = update_stats(fitness_values, population)
    
    # Calculate probability of every parent being selected 
    selection_prob = np.array( [ 2*i/(M*(M+1)) for i in range(1,M+1) ] )

    max_sols = 1000*dataset.shape[0]  # 1000*number of jobs  
    i = 0   # Counter
    Pm = Pminit  # Set mutation probability equal to initial
    
    while i < max_sols:        
        
        # Select position of the two parents
        select_p = np.random.choice(M, p = selection_prob) # Select using weight
        select_u = np.random.randint(M) # Select uniformly       
        
        # Store the two childen (same as parents for now)
        child1, child2 = population[select_p], population[select_u]
        
        # Apply crossover
        if random.random() < Pc:  
            child1, child2 = crossover(child1, child2)
            
        # Mutate childen    
        if random.random() < Pm : 
            child1 = swift_mutation(child1)
            child2 = swift_mutation(child2)
                
        # Choose randomly one of the children
        if random.randint(0, 1):
            child = np.copy(child1)
        else:
            child = np.copy(child2)
            
        # Choose randomly 1 element to delete with fitness below median
        x = random.randint(0, len(fitness_values)//2) 
        
        # Replace old sequence with child
        population[x] = child
        
        # Insert fitness value as well
        fitness_values[x] = makespan(dataset[child])
        # Update population statistics
        
        fitness_values, population = \
            update_stats(fitness_values, population)
        
        Pm = theta*Pm
        if np.min(fitness_values) / np.mean(fitness_values) > D : Pm = Pminit

        i += 1 # Increase counter
    
    # Return best makespan and its sequence
    return fitness_values[-1], population[-1]





def solve_GA(dataset, M = 30, Pc = 1, Pminit = 0.8, theta = 0.99, D = 0.95):
    """
    This function solves GA function 30 times,
    setting different random seed each time. 
    Then, it collects all computed makespans
    and the execution time of every GA call.
    
    :modules used: numpy as np, timeit, random
    :returns: array (30 makespans), float (average execution time of solutions)
    """
    # Initialize empty lists
    all_makespans = []  
    times = []
    
    for i in range(0,30):
        
        # Set random seed
        random.seed(i) 
        
        # Start timer
        start = timeit.default_timer()  
        
        # Compute best makespan
        mk, _ = GA(dataset, M, Pc, Pminit, theta, D) 
        
        # Stop timer
        stop = timeit.default_timer()   
        
        # Append best makespan and execution time
        all_makespans.append( mk )  
        times.append(stop - start)
    
    # Convert to arrays
    all_makespans = np.array(all_makespans) 
    times = np.array(times)
    
    return all_makespans, np.average(times)




# Execute GA for each dataset and save solution to DataFrame
#=============================================================
# Initialize dataframes
GA_df = pd.DataFrame()
GA_times = pd.DataFrame()

# Execute solve_GA and save results into DataFrames
start = timeit.default_timer() # Start timer
GA_df.loc[:, 'car1'], GA_times.loc['time', 'car1_time'] = solve_GA(car1)
GA_df.loc[:, 'car6'], GA_times.loc['time', 'car6_time'] = solve_GA(car6)
GA_df.loc[:, 'reC05'], GA_times.loc['time', 'reC05_time'] = solve_GA(reC05)
GA_df.loc[:, 'reC07'], GA_times.loc['time', 'reC07_time'] = solve_GA(reC07)
GA_df.loc[:, 'reC19'], GA_times.loc['time', 'reC19_time'] = solve_GA(reC19)
stop = timeit.default_timer()   # Stop timer
ga_overall_time = stop - start

print('\nGA - min makespan:\n\n', GA_df.min())
print('\nGA - mean makespan:\n\n', GA_df.mean())
print('\nGA - max makespan:\n\n', GA_df.max())
print('\nGA - std of makespan:\n\n', GA_df.std())
print('\nAverage execution times using GA:\n', GA_times)



# Store variables for later use
#=============================================================
# Save into pickles
f = open('GA_df.pckl', 'wb')
pickle.dump(GA_df, f)
f.close()
f = open('GA_times.pckl', 'wb')
pickle.dump(GA_times, f)
f.close()

# Load from pickles
f = open('GA_df.pckl', 'rb')
GA_df = pickle.load(f)
f.close()
f = open('GA_times.pckl', 'rb')
GA_times = pickle.load(f)
f.close()
