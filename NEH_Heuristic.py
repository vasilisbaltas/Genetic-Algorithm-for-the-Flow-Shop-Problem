# -*- coding: utf-8 -*-

import numpy as np
import random
import pandas as pd
import timeit
import pickle
from Fundamental_Calculation_Functions import makespan


car1 = pd.read_csv('car1.csv').values
car6 = pd.read_csv('car6.csv').values
reC05 = pd.read_csv('reC05.csv').values
reC07 = pd.read_csv('reC07.csv').values
reC19 = pd.read_csv('reC19.csv').values



def NEH(dataset):
    """
    This function implements the NEH heuristic algorithm.
    
    :modules used: numpy as np
    :returns: int (makespan), np.array (jobs)
    """
    
    
    n = dataset.shape[0]  # Number of jobs
    
    # Step 1: calculate sum of job i on every machine and store on T
    T = np.zeros(n, dtype=int)   # Initialize
    T = dataset.sum(axis = 1)
    
    # Step 2: arrange the jobs in descending order of T
    jobs = np.argsort(-T)  # np.argsort returns indexes
    
    
    # Step 3: find optimal partial sequence
    # Take first two jobs
    ds1 = np.take(dataset,[ jobs[0], jobs[1]], axis=0)
    ds2 = np.take(dataset,[ jobs[1], jobs[0]], axis=0)
    
    # Pick best makespan among these two sequences
    if makespan(ds1) > makespan(ds2):
        jobs[0:2] = np.take(jobs, [1,0])  

    i = 2  # 3rd position
    
    # Step 4: Find the best position of the next job
    while True:
        # Try all possible sequences for the next job
        # and keep the best one
        best_seq = np.zeros(i+1, dtype=int)
        best_seq[0] = jobs[i]
        best_seq[1:(i+1)] = jobs[0:i]
        best_makespan = makespan( np.take(dataset, best_seq, axis = 0) )
        
        test_seq = np.zeros((i+1), dtype=int)
        for j in range(1,(i+1)):
            test_seq[0:j] = jobs[0:j]
            test_seq[j] = jobs[i]
            test_seq[(j+1):(i+1)] = jobs[j:i]
            
            test_makespan = makespan( np.take(dataset, test_seq, axis = 0))
            if best_makespan > test_makespan :
                best_makespan = test_makespan
                best_seq = np.copy(test_seq)
            
        jobs[0:(i+1)] = best_seq[0:(i+1)]    
        # Step 5: Stop if all jobs are examined
        if n == (i+1):
            break
        else:
            i += 1
            
    
    return makespan(dataset[jobs]), jobs


# Execute NEH for each dataset and save solution to DataFrame
#=============================================================
# Initialize dataframes
neh_df = pd.DataFrame()
neh_times = pd.DataFrame()

# Insert neh solutions and computational times to dataframes
# Car1
start = timeit.default_timer()  
neh_df.loc['makespan', 'car1'], _ = NEH(car1)
stop = timeit.default_timer()   
neh_times.loc['time', 'car1'] = stop - start

# Car6
start = timeit.default_timer()  
neh_df.loc['makespan', 'car6'], _ = NEH(car6)
stop = timeit.default_timer()   
neh_times.loc['time', 'car6'] = stop - start

# reC05
start = timeit.default_timer()  
neh_df.loc['makespan', 'reC05'], _ = NEH(reC05)
stop = timeit.default_timer()   
neh_times.loc['time', 'reC05'] = stop - start

# reC07
start = timeit.default_timer()  
neh_df.loc['makespan', 'reC07'], _ = NEH(reC07)
stop = timeit.default_timer()   
neh_times.loc['time', 'reC07'] = stop - start

# reC19
start = timeit.default_timer()  
neh_df.loc['makespan', 'reC19'], _ = NEH(reC19)
stop = timeit.default_timer()   
neh_times.loc['time', 'reC19'] = stop - start

print('\nMakespan computed with NEH:\n', neh_df)
print('\nExecution times using NEH: \n', neh_times)
