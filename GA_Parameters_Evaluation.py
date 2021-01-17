# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import pickle
from time import timeit
from GA import solve_GA



#     Testing different parameters in Genetic Algorithm
#==============================================================================


car1 = pd.read_csv('car1.csv').values
car6 = pd.read_csv('car6.csv').values
reC05 = pd.read_csv('reC05.csv').values
reC07 = pd.read_csv('reC07.csv').values
reC19 = pd.read_csv('reC19.csv').values




# Store the parameters in lists
    
M = [5, 10, 20, 50, 100]    # population size
Pc = [0, 0.5, 0.7, 0.9]     # crossover probability
Pminit = [0, 0.2, 0.4, 0.6, 1]  # initial mutation probability
D = [0, 0.5, 1]     # threshold parameter



def M_parameters(dataset, M):
    """
    This function takes as input a dataset with
    job times on every machine and returns a 
    dataframe with all 30 GA makespan solutions 
    for each one of the different population 
    sizes, along with average times.
    
    :returns: pd.dataframe (makespans), dictionary (exeqution times)
    """
    # Create df to store makespan values for different
    df = pd.DataFrame(columns = M)
    # Initialize dictionary to store execution times
    times = {}
    for m in M: # iterate over population sizes
        mk, time = solve_GA(dataset, M = m)
        df.loc[:, m] = mk  # Store makespans on new df column
        times[m] = round(time, 3)   # Round and store execution time
     
    return df, times




def Pc_parameters(dataset, Pc):
    """
    Computes all 30 GA makespan solutions for each one 
    of the different crossover probability values in Pc.
    
    :returns: pd.dataframe (makespans), dictionary (exeqution times)
    """
    # Create df to store makespan values
    df = pd.DataFrame(columns = Pc)
    # Initialize dictionary to store execution times
    times = {}
    for pc in Pc: # iterate over Pc values
        mk, time = solve_GA(dataset, Pc = pc)
        df.loc[:, pc] = mk  # Store makespans on new df column
        times[pc] = round(time, 3)   # Round and store execution time
     
    return df, times




def Pminit_parameters(dataset, Pminit):
    """
    Computes all 30 GA makespan solutions for each one 
    of the different initial mutation probability values in Pminit.
        
    :returns: pd.dataframe(makespans), dictionary (exeqution times)
    """
    # Create df to store makespan values
    df = pd.DataFrame(columns = Pminit)
    # Initialize dictionary to store execution times
    times = {}
    for pminit in Pminit: # iterate over Pminit values
        mk, time = solve_GA(dataset, Pminit = pminit)
        df.loc[:, pminit] = mk  # Store makespans on new df column
        times[pminit] = round(time, 3)   # Round and store execution time
     
    return df, times




def D_parameters(dataset, D):
    """
    Computes all 30 GA makespan solutions for each one 
    of the different threshold parameter values in D.
        
    :returns: pd.dataframe (makespans), dictionary (exeqution times)
    """
    # Create df to store makespan values
    df = pd.DataFrame(columns = D)
    # Initialize dictionary to store execution times
    times = {}
    for d in D: # iterate over D values
        mk, time = solve_GA(dataset, D = d)
        df.loc[:, d] = mk  # Store makespans on new df column
        times[d] = round(time, 3)   # Round and store execution time
     
    return df, times




#====== Run the algorithm for every parameter =============
    
# Measure time needed to compute all variables 
start_all = timeit.default_timer()  

#--- Makespan and execution times for different M values for each car -----
m_car1, m1_times = M_parameters(car1, M)
m_car6, m6_times = M_parameters(car6, M)
m_reC05, m05_times = M_parameters(reC05, M)
m_reC07, m07_times = M_parameters(reC07, M)
m_reC19, m19_times = M_parameters(reC19, M)

#--- Makespan and execution times for different Pc values for each car -----
pc_car1, pc1_times = Pc_parameters(car1, Pc)
pc_car6, pc6_times = Pc_parameters(car6, Pc)
pc_reC05, pc05_times = Pc_parameters(reC05, Pc)
pc_reC07, pc07_times = Pc_parameters(reC07, Pc)
pc_reC19, pc19_times = Pc_parameters(reC19, Pc)

#-- Makespan and execution times for different Pminit values for each car ---
pminit_car1, pminit1_times = Pminit_parameters(car1, Pminit)
pminit_car6, pminit6_times = Pminit_parameters(car6, Pminit)
pminit_reC05, pminit05_times = Pminit_parameters(reC05, Pminit)
pminit_reC07, pminit07_times = Pminit_parameters(reC07, Pminit)
pminit_reC19, pminit19_times = Pminit_parameters(reC19, Pminit)

#--- Makespan and execution times for different D values for each car -----
d_car1, d1_times = D_parameters(car1, Pminit)
d_car6, d6_times = D_parameters(car6, Pminit)
d_reC05, d05_times = D_parameters(reC05, Pminit)
d_reC07, d07_times = D_parameters(reC07, Pminit)
d_reC19, d19_times = D_parameters(reC19, Pminit)


#====== Stop timer ==============
stop_all = timeit.default_timer() 
time_all = stop_all - start_all
print('Time needed:', time_all)
# Time needed: 16360 seconds ~ 4.5 hours




# Visualization for different parameters
#==================================================================


def create5subplots(d1, d2, d3, d4, d5, title,
                    suptitle, xlabel, ylabel = 'Makespan'):
    '''
    This function uses matplotlib to create a figure with 5 subplots.
    It is used to visualize the effect of only one parameter on 
    every instance (car).
    '''
    
    
    # Create figure
    fig = plt.figure(facecolor = '#FCF4DC')
    # Create subplot axes
    ax1 = plt.subplot2grid((4,6), (0,0), rowspan = 2, colspan = 2)
    ax2 = plt.subplot2grid((4,6), (0,2), rowspan = 2, colspan = 2)
    ax3 = plt.subplot2grid((4,6), (0,4), rowspan = 2, colspan = 2)
    ax4 = plt.subplot2grid((4,6), (2,0), rowspan = 2, colspan = 3)
    ax5 = plt.subplot2grid((4,6), (2,3), rowspan = 2, colspan = 3)
    # Give color to boxplots
    red_square = dict(markerfacecolor='r', marker='s', markersize=4)
    # Plot data
    d1.plot(kind = 'box', ax=ax1, patch_artist=True, flierprops=red_square)
    d2.plot(kind = 'box', ax=ax2, patch_artist=True, flierprops=red_square)
    d3.plot(kind = 'box', ax=ax3, patch_artist=True, flierprops=red_square)
    d4.plot(kind = 'box', ax=ax4, patch_artist=True, flierprops=red_square)
    d5.plot(kind = 'box', ax=ax5, patch_artist=True, flierprops=red_square)
    # Add titles to subplots
    ax1.set_title(title[0], fontstyle = 'italic')
    ax2.set_title(title[1], fontstyle = 'italic')
    ax3.set_title(title[2], fontstyle = 'italic')
    ax4.set_title(title[3], fontstyle = 'italic')
    ax5.set_title(title[4], fontstyle = 'italic')
    # Set main title and labels
    plt.suptitle(suptitle, fontfamily = 'serif', fontsize=14)   
    # add x-label to subplot
    ax4.set_xlabel(xlabel, fontfamily = 'serif')
    ax5.set_xlabel(xlabel, fontfamily = 'serif')
    # add y-label to subplot
    ax1.set_ylabel(ylabel, fontfamily = 'serif')
    ax4.set_ylabel(ylabel, fontfamily = 'serif')
    
    # Make it more beautiful and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(xlabel+'_effect.pdf', facecolor=fig.get_facecolor() )



# Declare the input variables
titles = ['Car 1 (11x5)', 'Car 6 (8x9)', 'reC05 (20x5)',
         'reC07 (20x10)', 'reC19 (30x10)']
xlabels = ['M','Pc','Pm initial','D']
suptitles = ['Effect of varying population size',
            'Effect of varying crossover probability',
            'Effect of varying initial mutation probability',
            'Effect of varying threshold parameter' ]


# Create plot for each instance to visualize the
# effect of varying the parameters (one at a time)
create5subplots(m_car1, m_car6, m_reC05, m_reC07, m_reC19,
                                titles, suptitles[0], xlabels[0])

create5subplots(pc_car1, pc_car6, pc_reC05, pc_reC07, pc_reC19,
                                titles, suptitles[1], xlabels[1])

create5subplots(pminit_car1, pminit_car6, pminit_reC05,
                pminit_reC07, pminit_reC19, titles, suptitles[2], xlabels[2])
             
create5subplots(d_car1, d_car6, d_reC05, d_reC07, d_reC19,
                                            titles, suptitles[3], xlabels[3])



#------------------------------------------------------------------------------


def create4subplots(d1, d2, d3, d4, car, xlabel, ylabel = 'Makespan'):
    '''
    This function uses matplotlib to create a figure with 4 subplots.
    It is used to visualize the effect of only all parameter on 
    only one instance.
    '''
    
    
    suptitle = 'Effect of varying GA parameters on '+ car
    # Create figure
    fig = plt.figure(facecolor = '#FCF4DC')
    # Create subplot axes
    ax1 = plt.subplot2grid((4,6), (0,0), rowspan = 2, colspan = 3)
    ax2 = plt.subplot2grid((4,6), (0,3), rowspan = 2, colspan = 3)
    ax3 = plt.subplot2grid((4,6), (2,0), rowspan = 2, colspan = 3)
    ax4 = plt.subplot2grid((4,6), (2,3), rowspan = 2, colspan = 3)
    # Give color to boxplots
    red_square = dict(markerfacecolor='r', marker='s', markersize=4)
    # Plot data
    d1.plot(kind = 'box', ax=ax1, patch_artist=True, flierprops=red_square)
    d2.plot(kind = 'box', ax=ax2, patch_artist=True, flierprops=red_square)
    d3.plot(kind = 'box', ax=ax3, patch_artist=True, flierprops=red_square)
    d4.plot(kind = 'box', ax=ax4, patch_artist=True, flierprops=red_square)
    # Set main title and labels
    plt.suptitle(suptitle, fontfamily = 'serif', fontsize=14)
    
    # add x-label to subplot
    ax1.set_xlabel(xlabel[0], fontfamily = 'serif')
    ax2.set_xlabel(xlabel[1], fontfamily = 'serif')
    ax3.set_xlabel(xlabel[2], fontfamily = 'serif')
    ax4.set_xlabel(xlabel[3], fontfamily = 'serif')
    
    # add y-label to subplot
    ax1.set_ylabel(ylabel, fontfamily = 'serif')
    ax3.set_ylabel(ylabel, fontfamily = 'serif')
    
    # Make it more beautiful and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig(car+'_all_parameters.pdf', facecolor=fig.get_facecolor() )


 # Use function to plot the effect   
create4subplots(m_car6, pc_car6, pminit_car6, d_car6, titles[1], xlabels)

                
# After analyzing the plots, we obtain some optimal parameter values
# and we repeat GA on reC19 using these values
GA_reC19 = pd.DataFrame()
GA_reC19.loc[:,'Default parameters'] = GA_df.loc[:,'reC19']
GA_reC19.loc[:,'Optimal parameters'] , _ = solve_GA(reC19, M=50, 
                                                   Pc=0.9, Pminit=0.6, D=0.4)

# Visualize the default vs the optimal parameters
fig = plt.figure(facecolor = '#FCF4DC')
red_square = dict(markerfacecolor='r', marker='s', markersize=4)
GA_reC19.plot(kind = 'box', patch_artist=True, flierprops=red_square,
                  medianprops={'linestyle': '-', 'linewidth': 5},
                               showmeans=True,  color={'medians': 'red'})
plt.suptitle('Visualizing GA results using default and optimal parameters',
                                             fontfamily = 'serif', fontsize=13)

plt.xlabel('reC19 instance', fontsize=13)
plt.ylabel('Makespan', fontsize=13)
plt.savefig('optimal.pdf', facecolor=fig.get_facecolor() )



