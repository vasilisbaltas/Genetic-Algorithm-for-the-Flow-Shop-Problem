# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
import timeit
import pickle
from Fundamental_Calculation_Functions import makespan


car1 = pd.read_csv('car1.csv').values
car6 = pd.read_csv('car6.csv').values
reC05 = pd.read_csv('reC05.csv').values
reC07 = pd.read_csv('reC07.csv').values
reC19 = pd.read_csv('reC19.csv').values




def random_search(dataset):
    """
    This function applies makespan algorithm to 
    dataset using 1000×𝑛 random solution evaluations.
    
    :modules used: numpy as np, random
    :returns: int (best makespan), np.array (best sequence)    
    """
    # Generate and evaluate one random solution
    x = np.arange(0, dataset.shape[0]) # Generate values (0...number of jobs)
    random.shuffle(x)   # Modify the sequence
    best_makespan = makespan(dataset[x])   
    best_sequence = np.copy(x)
    
    # Declare maximum solution evaluations and counter
    max_sols = 1000*dataset.shape[0] # 1000*number of jobs
    i = 0   # Counter
    while i < max_sols:
        random.shuffle(x)
        temp_makespan = makespan(dataset[x])            
        if best_makespan > temp_makespan:
            best_makespan = temp_makespan
            best_sequence = np.copy(x)
        i += 1

    return best_makespan, best_sequence





def solve_RS(dataset):
    """
    This function solves random_search function 30 times, 
    setting different random seed each time. Then, it 
    collects all computed makespans and the execution time
    of every RS call.
    
    :modules used: numpy as np, timeit, random
    :returns: array (30 makespans), float (execution time of solutions)
    """
    # Initialize empty lists
    all_makespans = []  
    times = []    
    for i in range(0,30):
        # Set random seed to shuffle in a different way on random_search
        random.seed(i)  
        # Start timer
        start = timeit.default_timer()  
        # Compute best makespan
        mk, _ = random_search(dataset)
        # Stop timer
        stop = timeit.default_timer()   
        # Append best makespan and execution time
        all_makespans.append( mk )  
        times.append( round(stop - start, 3) )
     
    # Convert to arrays    
    all_makespans = np.array(all_makespans) 
    times = np.array(times)

    return all_makespans, times





# Execute solve_RS for each dataset and save solution to DataFrame
#=================================================================
RS_df = pd.DataFrame()
RS_df.loc[:, 'car1'], RS_df.loc[:, 'car1_time'] = solve_RS(car1)
RS_df.loc[:, 'car6'], RS_df.loc[:, 'car6_time'] = solve_RS(car6)
RS_df.loc[:, 'reC05'], RS_df.loc[:, 'reC05_time'] = solve_RS(reC05)
RS_df.loc[:, 'reC07'], RS_df.loc[:, 'reC07_time'] = solve_RS(reC07)
RS_df.loc[:, 'reC19'], RS_df.loc[:, 'reC19_time'] = solve_RS(reC19)

# Get min, max and std of each car
RS_df.iloc[:,::2].min()
RS_df.iloc[:,::2].max()
RS_df.iloc[:,::2].mean()
RS_df.iloc[:,::2].std()

# Get average computational time
RS_df.iloc[:,1::2].mean()

# Store variables for later use
#=============================================================
# Save into pickles
f = open('RS_df.pckl', 'wb')
pickle.dump(RS_df, f)
f.close()
# Load from pickles
f = open('RS_df.pckl', 'rb')
RS_df = pickle.load(f)
f.close()





