# -*- coding: utf-8 -*-

import numpy as np
import random
import pandas as pd



car1 = pd.read_csv('car1.csv').values
car6 = pd.read_csv('car6.csv').values
reC05 = pd.read_csv('reC05.csv').values
reC07 = pd.read_csv('reC07.csv').values
reC19 = pd.read_csv('reC19.csv').values



def crossover(P1, P2):
    """
    This function is used inside general genetic algorithm function (GA).
    Executes 2 point crossover on the input 
    arrays (parents), while the crossover points X, Y
    are chosen randomly. The childen contain the 
    pre-X section of the one parent and the post-Y 
    section of the other parent. The gaps are 
    filled randomly by the unused elements.
    
    :modules used: numpy as np, random
    :returns: 2 arrays (children)
    """


    size = P1.shape[0]  # Size of input array
    child1 = np.array([-1]*size) # Set it equal to -1 because 0 is a job
    child2 = np.array([-1]*size) # and there will be confusion
    
    
    # Choose 2 crossover points
    X = random.randint(1, size)
    Y = random.randint(1, size - 1)
    if Y >= X:
        Y += 1
    else: # Swap the two points
        X, Y = Y, X
        
        
    
    #-- Generating the first child --   
    child1[0:X] = P1[0:X] # Pre-X of the one parent
    # Check if elements of post-Y are also in pre-X and create boolean mask
    
    mask = np.append( [[False]*Y], 
                     np.isin(P2[Y:size], P1[0:X], invert = True))
    
    # Assign P2 post-Y unique values to child
    child1[mask] = P2[mask]    
    
    # setdiff1d returns the sorted, unique values in P1 that are not in child1
    not_in_child = np.setdiff1d(P1, child1)
    
    # Then rearrange it randomly 
    random.shuffle(not_in_child)
    
    # and assign it to child where values equal to -1
    child1[np.where(child1 == -1)] = not_in_child     
    
    # Same as above, for child2 this time
    child2[0:X] = P2[0:X]
    mask = np.append( [[False]*Y], 
                     np.isin(P1[Y:size], P2[0:X], invert = True))
    child2[mask] = P1[mask]  
    not_in_child = np.setdiff1d(P1, child2)
    random.shuffle(not_in_child)
    child2[np.where(child2 == -1)] = not_in_child
    
    
    
    return child1, child2
                 



def swift_mutation(child):
    """ 
    This function is used inside GA function.
    Shifts one element of the list (chosen
    randomly) a random number of places to 
    the right or left.
    
    :moduled used: numpy as np, random
    :returns: np.array (shifted child)  
    """
    
    
    size = child.shape[0]
    
    # Initialize the output array
    shifted_child = np.zeros(size, dtype = int)
    
    # Choose one random element (but not first or last)
    element_pos = random.randint(1, size-2)
    
    # Choose randomly left or right
    if random.randint(0, 1): # If left 
        # Choose the new position of the element
        x = random.randint(0, element_pos)
        
        # Change the position of the element
        shifted_child[0:x] = child[0:x]
        shifted_child[x] = child[element_pos]
        shifted_child[(x+1):(element_pos+1)] = child[x:element_pos]
        shifted_child[(element_pos+1):size] = child[(element_pos + 1):size]
    
    else: # Else if right
        # Choose the new position of the element
        x = random.randint(element_pos+1, size-1)
        # Change the position of the element
        shifted_child[0:element_pos] = child[0:element_pos]
        shifted_child[element_pos : x] = child[(element_pos+1):(x+1)]
        shifted_child[x] = child[element_pos]
        shifted_child[(x+1):size] = child[(x+1):size]
        
        
    return shifted_child





def update_stats(fitness_values, population):
    """
    Used inside GA function. Takes as input
    the 2 numpy arrays (population and its
    fitness_values) and sorts them descending
    according to fitness values.
    
    :modules used: numpy as np
    :returns: 2 arrays
    """
    order = np.argsort(-fitness_values)
   
    return fitness_values[order], population[order]