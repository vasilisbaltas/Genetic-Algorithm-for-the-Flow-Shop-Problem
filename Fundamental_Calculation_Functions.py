# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
import pickle



#==============================================================================
# Reading and cleaning data from txt file
#==============================================================================

with open('flowshop_bman73701.txt', 'r') as f:
 lines = f.readlines()
 car1 = []
 car6 = []
 reC05 = []
 reC07 = []
 reC19 = []
 
 for num, row in enumerate(lines):
     
  if (num>=41 and num<=51): # Row 41 until 51 in txt file
    car1.append(row.split())
  if (num>=59 and num<=66):
    car6.append(row.split())
  if (num>=74 and num<=93):
    reC05.append(row.split())
  if (num>=101 and num<=120):
    reC07.append(row.split())
  if (num>=128 and num<=157):
    reC19.append(row.split())
    
    
# Convert to int np.array, delete unwanted columns
car1 = np.array(car1)
car1 = np.delete(car1, np.s_[::2], 1).astype(np.int)
car6 = np.array(car6)
car6 = np.delete(car6, np.s_[::2], 1).astype(np.int)
reC05 = np.array(reC05)
reC05 = np.delete(reC05, np.s_[::2], 1).astype(np.int)
reC07 = np.array(reC07)
reC07 = np.delete(reC07, np.s_[::2], 1).astype(np.int)
reC19 = np.array(reC19)
reC19 = np.delete(reC19, np.s_[::2], 1).astype(np.int)
del row, lines, num # Delete unwanted variables



pd.DataFrame(car1).to_csv('car1.csv', encoding = 'utf-8',index=False)
pd.DataFrame(car6).to_csv('car6.csv', encoding = 'utf-8',index=False)
pd.DataFrame(reC05).to_csv('reC05.csv', encoding = 'utf-8',index=False)
pd.DataFrame(reC07).to_csv('reC07.csv', encoding = 'utf-8',index=False)
pd.DataFrame(reC19).to_csv('reC19.csv', encoding = 'utf-8',index=False)





def makespan(dataset):
    
  ##  This function computes the makespan of the dataset
  ##  (where rows are jobs and columns are the machines)
  ##  for this specific sequence of jobs.
  ##  :modules used: numpy as np
  ##  :returns: int (makespan)
    
   rows, cols = dataset.shape
   CSkj = np.zeros((rows,cols), dtype = int) # Create array full of zeros
   
   
# Compute first row of makespan table
   CSkj[0, :] = np.cumsum(dataset[0,:])
# Compute first column of makespan table
   CSkj[:, 0] = np.cumsum(dataset[:,0])
  
   for j in range(1, cols):
     for i in range(1, rows):
        CSkj[i,j] = max(CSkj[i-1,j], CSkj[i,j-1]) + dataset[i,j]
        
   return CSkj[-1,-1]



















