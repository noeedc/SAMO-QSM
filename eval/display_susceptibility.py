"""
Script: QSM Susceptibility Visualization

Description:
This script reads region of interest (ROI) susceptibility data from a CSV file and visualizes mean susceptibility values for multi-resolution data.
It utilizes the matplotlib, pandas, and seaborn libraries for data manipulation and visualization.

Author: Noée Ducros-Chabot
Date: December 14th 2023

Dependencies:
- matplotlib
- pandas
- seaborn

Usage:
1. Ensure the required libraries are installed: matplotlib, pandas, seaborn.
   You can install them using: pip install matplotlib pandas seaborn

2. Update the 'suscep_csv_file' variable with the path to your susceptibility data CSV file.

3. Run the script to generate boxplots illustrating mean susceptibility values for different methods and resolutions.

Note: This script assumes a specific structure in the input CSV file, including columns like 'Inversion', 'Res', 'nOri', 'Label', and 'average'.

"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Script to display mean susceptibility values for multi-resolution data
if __name__ == "__main__":
    suscep_csv_file = '/home/magic-chusj-2/Documents/E2022/DATA/ex_ds_dataset/data/1.05mm/results/whole_mean_susceptibilities.csv'
    df = pd.read_csv(suscep_csv_file, index_col=0)
    
    # Sorting the df
    key = {
        'iLSQR': 1,
        'NDI': 2,
        'Star-QSM': 3,
        'TKD': 4,
        'FANSI': 5,
        'MEDI': 6,
        'ILR-mCOSMOS': 7,
        'SAMO-QSM': 8,
        'COSMOS': 9
    }

    # Add a temporary column for sorting
    df['SortKey'] = df['Inversion'].map(key)

    # Sort the DataFrame based on the temporary column
    df = df.sort_values(by=['nOri', 'SortKey'])

    # Drop the temporary column
    df.drop(columns=['SortKey'], inplace=True)
  
   # Plot as boxplot 
    sns.set_style("ticks",{'axes.grid' : True})
    sns.set_palette("husl", n_colors = df['Res'].nunique() )

    # Plot as subplots
    f, axes = plt.subplots(1, 5)
    small_font = 15 
    large_font = 20
    for i, (label, ax) in enumerate(zip(df['Label'].unique(), axes.ravel()[:len(df['Label'].unique())])):
        target = df.query((f'nOri==5 & Inversion == "COSMOS" & Label == "{label}"'))['average'].values[0]
        ax.axhline(target, color = 'red', linewidth = '1', alpha=0.7)
        
        sns.boxplot(y="average", x= "Method", data=df[df['Label']==label], ax=ax, hue = 'Res')
        # ax.tick_params(labelrotation=68)
        ax.tick_params(labelsize=13) 
        ax.set_xticklabels(ax.get_xticklabels(), rotation=80)  # Rotate x-axis tick labels
        if i == 0 :
            ax.set_ylabel('Mean susceptibility value (ppm)', fontsize=small_font)
        else :
            ax.set_ylabel('')  # Remove the ylabel for all other subplots
        ax.set_yticklabels(ax.get_yticklabels(), rotation=45)
        ax.set_xlabel('')
        ax.set_title(label.capitalize(), fontsize=large_font)
        ax.axvline(ax.get_xticks()[-4]+0.5, color = 'darkgrey', linestyle = 'dashed', linewidth = '1')
        ax.axvline(ax.get_xticks()[-7]+0.5, color = 'darkgrey', linestyle = 'dashed', linewidth = '1')
        ax.axvline(ax.get_xticks()[-10]+0.5, color = 'darkgrey', linestyle = 'dashed', linewidth = '1')
    plt.tight_layout()
    # f.suptitle('Susceptibility means for ROIs', size= large_font, fontweight="bold") # Uncomment if want title
    plt.subplot_tool()
    f.supxlabel('Method', fontsize=small_font)
    plt.show()