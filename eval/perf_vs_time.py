"""
Script: Evaluation of Performance vs Time

Description:
This script reads performance metrics from a CSV file and visualizes the evaluation of performance metrics 
(1-NRMSE and XSIM) across different methods and orientations. The script includes functionality to 
display the corresponding time values for each method based on the number of orientations and resolution.

Author: Noée Ducros-Chabot
Date: December 14th 2023

Dependencies:
- numpy
- pandas
- seaborn
- matplotlib
- re

Usage:
1. Ensure the required libraries are installed: numpy, pandas, seaborn, matplotlib.
   You can install them using: pip install numpy pandas seaborn matplotlib

2. Update the 'results_csv' variable with the path to your performance metrics CSV file.

3. Modify the 'query_condition' variable if you want to filter the data based on specific conditions.

4. Run the script to generate a line plot showcasing the performance metrics (1-NRMSE and XSIM) for different methods.

Note: This script assumes a specific structure in the input CSV file, including columns like 'nOri', 'Method', 'Combination', '1-NRMSE', and 'XSIM'.
"""

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re 
import pandas as pd
from eval.calculate_metrics import extract_resolution

def format_metric(name):
    if 'mm' not in name:
        orientation, method = name.split('-')[0], '-'.join(name.split('-')[1:])
        formatted = f'{orientation} orientations.\n{method}'
    else : 
        parts = name.split('-')
        orientation, res, method = parts[0], parts[1], '-'.join(parts[2:])
        formatted = f'{orientation} orientations. {res}.\n{method}'
    return formatted

def get_time_from_method(method):
    n = int(method.split()[0])
    res = float(extract_resolution(method).replace('mm',''))

    if res == 1 or res == 1.05:
        time = n*(7+37/60)
    elif res == 1.9:
        time = (7+37/60) + (n-1)*(2+45/60)
    elif res == 2.4:
        time = (7+37/60) + (n-1)*(1+57/60)
    return time

def time_from_string(time):
    min = int(time)
    sec = int((time%1)*60)
    time_str = f"{min}:{sec}\nminutes"

    return time_str

if __name__ == "__main__":
    results_csv = '/home/magic-chusj-2/Documents/E2022/DATA/ex_ds_dataset/data/1.05mm/results/image_quality_metrics.csv'
    df = pd.read_csv(results_csv, index_col=0)

    query_condition = '((tkd == 0.20) & (w == 0.5)) | ((Method == "HR") & (nOri != 1))| ((nOri == 1) & (Inversion == "Star-QSM"))'
    df = df.query(query_condition)
    print(df.groupby(['nOri', 'Method']).mean())

    # Change method name for Inversion method in single orientation data
    df.loc[df['nOri'] == 1, 'Method'] = df.loc[df['nOri'] == 1, 'Inversion']

    # Change values from 'pred' to 'SAMO-QSM' in the 'method' column
    df.loc[df['Method'] == 'ILR', 'Method'] = 'ILR-mCOSMOS'
    df.loc[df['Method'] == 'pred', 'Method'] = 'SAMO-QSM'
    df.loc[df['Method'] == 'HR', 'Method'] = 'COSMOS'

    if 'Res' in df.keys():
        df['n_Method'] = df['nOri'].astype(str) + '-' + df['Res']+'-' + df['Method']
    else :
        df['n_Method'] = df['nOri'].astype(str) +'-' + df['Method']
    df.insert(0, 'n_Method', df.pop('n_Method'))

    fr = False
    metrics = list(df.columns.values)[-4:]
    target_PSNR = df['PSNR'].where(df['PSNR']!= float('inf')).max() + 2
    df['PSNR'] = df['PSNR'].replace(np.inf, target_PSNR) # Replace inf with target value (for plotting purposes)
    target = [1, 1, target_PSNR, 2]
    sOri_Mean = df[df['nOri'] == 1][metrics].mean().values
    sOri_Mins = df[df['nOri'] == 1][metrics].min().values
    sOri_Max = df[df['nOri'] == 1][metrics].max().values

    ## Data plotting:
    combinations = df['Combination'].unique()[1:]

    #sns.set_style("ticks", font_scale=1.5, {'axes.grid' : True})
    sns.set(style="ticks", font_scale=1, rc={"axes.grid": True})
    color_palette = sns.color_palette("husl", n_colors=3)
    sns.set_palette(color_palette)
    # sns.set(font_scale=1.5)

    # Apply the formatting function to the Metric column
    df['n_Method'] = df['n_Method'].apply(format_metric)

    stripped_df = pd.melt(df, id_vars=['n_Method'], value_vars=['1-NRMSE', 'XSIM'], var_name='Metric', value_name='Value')

    t_order = 'time'
    if t_order == 'time':
        time_dict = {
            '1 orientations.\nStar-QSM': 7,
            '3 orientations.\nILR-mCOSMOS': 10.5,
            '3 orientations.\nSAMO-QSM': 10.51,
            '4 orientations.\nILR-mCOSMOS': 12.25,
            '4 orientations.\nSAMO-QSM': 12.25,
            '5 orientations.\nILR-mCOSMOS': 14,
            '5 orientations.\nSAMO-QSM': 14,
            '3 orientations.\nCOSMOS': 21,
            '4 orientations.\nCOSMOS': 28,
            '5 orientations.\nCOSMOS': 35
            }
        stripped_df['Time'] = stripped_df['n_Method'].map(get_time_from_method)
        stripped_df = stripped_df.sort_values(by='Time')
        stripped_df['Time'] = stripped_df['Time'].map(time_from_string)

    elif t_order == 'nOri':
        orientation_list = [
        '1 orientations.\nStar-QSM',
        '3 orientations.\nILR-mCOSMOS',
        '3 orientations.\nSAMO-QSM',
        '3 orientations.\nCOSMOS',
        '4 orientations.\nILR-mCOSMOS',
        '4 orientations.\nSAMO-QSM',
        '4 orientations.\nCOSMOS',
        '5 orientations.\nILR-mCOSMOS',
        '5 orientations.\nSAMO-QSM',
        '5 orientations.\nCOSMOS'
        ]
        stripped_df['n_Method'] = pd.Categorical(stripped_df['n_Method'], categories=orientation_list, ordered=True)
        
    font = 20
    font_small = 18 
    plt.figure(figsize=(20, 8))
    plt.tick_params(labelsize=font)
    ax = sns.lineplot(x= 'n_Method', y = 'Value', hue = 'Metric', data=stripped_df, err_style=None)
    sns.scatterplot(x= 'n_Method', y = 'Value', hue = 'Metric', data=stripped_df,  marker = 'X', legend=False)
    ax.set_ylabel('Value', fontsize=font)
    ax.set_xlabel('Method',  weight = 'bold', fontsize=font)
    # Rotate only the x-axis ticks
    plt.xticks(rotation=45, fontsize=font_small)
    # for i in [0,2,4,6,7,8]:
    #     ax.axvline(ax.get_xticks()[i]+0.5, color = color_palette[2], linestyle = 'dashed', linewidth = '1')

    # Add a second x-axis on top with time values
    second_ax = ax.twiny()
    second_ax.set_xlim(ax.get_xlim())
    
    new_ticks = [0] + [x + 0.5 for x in ax.get_xticks()[1:-3:2]] + ax.get_xticks()[-3:]
    second_ax.set_xticks(new_ticks)
    time_labels = stripped_df['Time'].unique()
    second_ax.set_xlabel('Time',  weight = 'bold', fontsize=font)
    second_ax.set_xticklabels(time_labels)
    second_ax.grid(False)
    #second_ax.grid(True, linestyle='dashed', linewidth=0.5, color='gray', alpha=0.7, zorder=0)  # Set z-order for gridlines
    plt.tick_params( labelsize=font_small)

    plt.tight_layout()
    plt.subplot_tool()
    plt.show()