# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import pickle


#======== VISUALIZATION: Comparison between algorithms ================



# Initialize dataframes which will help us plot
plot_car1 = pd.DataFrame()
plot_car6 = pd.DataFrame()
plot_reC05 = pd.DataFrame()
plot_reC07 = pd.DataFrame()
plot_reC19 = pd.DataFrame()



# Store values into the dataframes
plot_car1.loc[:,'RS'] = RS_df.loc[:,'car1']
plot_car1.loc[:,'NEH'] = neh_df.loc['makespan','car1']
plot_car1.loc[:,'GA'] = GA_df.loc[:,'car1']

plot_car6.loc[:,'RS'] = RS_df.loc[:,'car6']
plot_car6.loc[:,'NEH'] = neh_df.loc['makespan','car6']
plot_car6.loc[:,'GA'] = GA_df.loc[:,'car6']

plot_reC05.loc[:,'RS'] = RS_df.loc[:,'reC05']
plot_reC05.loc[:,'NEH'] = neh_df.loc['makespan','reC05']
plot_reC05.loc[:,'GA'] = GA_df.loc[:,'reC05']

plot_reC07.loc[:,'RS'] = RS_df.loc[:,'reC07']
plot_reC07.loc[:,'NEH'] = neh_df.loc['makespan','reC07']
plot_reC07.loc[:,'GA'] = GA_df.loc[:,'reC07']

plot_reC19.loc[:,'RS'] = RS_df.loc[:,'reC19']
plot_reC19.loc[:,'NEH'] = neh_df.loc['makespan','reC19']
plot_reC19.loc[:,'GA'] = GA_df.loc[:,'reC19']


# Create function to plot
def plot_comparison(p1, p2, p3, p4, p5):
    """
    This function plots in 5 subplots the RS, 
    NEH and GA results on all instances.
    """
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

    # Plot boxplots
    p1.plot(kind = 'box', ax=ax1, patch_artist=True, flierprops=red_square)
    p2.plot(kind = 'box', ax=ax2, patch_artist=True, flierprops=red_square)
    p3.plot(kind = 'box', ax=ax3, patch_artist=True, flierprops=red_square)
    p4.plot(kind = 'box', ax=ax4, patch_artist=True, flierprops=red_square)
    p5.plot(kind = 'box', ax=ax5, patch_artist=True, flierprops=red_square)
    
    # Add titles to subplots
    ax1.set_title('Car 1 (11x5)', fontstyle = 'italic')
    ax2.set_title('Car 6 (8x9)', fontstyle = 'italic')
    ax3.set_title('reC05 (20x5)', fontstyle = 'italic')
    ax4.set_title('reC07 (20x10)', fontstyle = 'italic')
    ax5.set_title('reC19 (30x10)', fontstyle = 'italic')
    
    # Set main title and labels
    plt.suptitle('Comparison between the algorithms',
                 fontfamily = 'serif', fontsize=14)   
    # add y-label to subplot
    ax1.set_ylabel('Makespan', fontfamily = 'serif')
    ax4.set_ylabel('Makespan', fontfamily = 'serif')
    
    # Make it more beautiful and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig('comparison.pdf', facecolor=fig.get_facecolor() )
#-----------------------------
    
# Use function to plot and save as pdf
plot_comparison(plot_car1, plot_car6, plot_reC05, plot_reC07, plot_reC19)




