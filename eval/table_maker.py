"""
Script: Image Quality Metrics Aggregation and Formatting

Description:
This script reads image quality metrics from a CSV file, filters the data based on specified criteria, and aggregates
mean and standard deviation values for each group (combination of 'nOri', 'Method', and optionally 'Res'). The script
then formats the results and saves them to a new CSV file for further analysis.

Author: Noée Ducros-Chabot
Date: December 14th 2023

Dependencies:
- pandas
- os

Usage:
1. Ensure the required libraries are installed: pandas.
   You can install them using: pip install pandas

2. Update the 'csv_file' variable with the path to your image quality metrics CSV file.

3. Modify the 'query' variable if you want to filter the data based on specific conditions.

4. Run the script to aggregate mean and standard deviation values for each group, format the results, and save them to a new CSV file.

Note: This script assumes a specific structure in the input CSV file, including columns like 'nOri', 'Method', 'Res', and image quality metrics.

"""

import pandas as pd 
import os

csv_file = '/home/magic-chusj-2/Documents/E2022/DATA/ex_ds_dataset/data/1.05mm/results/image_quality_metrics.csv'
df = pd.read_csv(csv_file, index_col=0)

# Filter only important data
query = '((tkd == 0.20) & (w == 0.5)) | (Method == "HR") | (nOri == 1)'
df = df.query(query)
df.loc[df['nOri'] == 1, 'Method'] = df.loc[df['nOri'] == 1, 'Inversion']
df['NRMSE'] = 1 - df['1-NRMSE']
df = df.drop(columns=['Inversion', '1-NRMSE'])

# Calculate mean and standard deviation for each group
metrics = df.columns[-4:].values.tolist()
metrics.insert(0, metrics.pop()) # Put NRMSE as first
method_mapping = {
    'HR': 'COSMOS',
    'ILR': 'ILR-mCOSMOS',
    'pred': 'SAMO-QSM'
}
df['Method'] = df['Method'].replace(method_mapping)

if 'Res' in df.keys():
    group_stats = df.groupby(['nOri', 'Method', 'Res'])[metrics].agg(['mean', 'std'])
else :
    group_stats = df.groupby(['nOri', 'Method'])[metrics].agg(['mean', 'std'])

# Create a new DataFrame to store formatted results
formatted_stats = pd.DataFrame()

# Loop through metrics and calculate mean ± std
for metric in metrics:
    formatted_mean_std = (
        group_stats[metric]['mean'].apply(lambda x: format(x, '.3g')) + 
        ' ± ' + 
        group_stats[metric]['std'].apply(lambda x: format(x, '.2g'))
    )
    formatted_stats[metric] = formatted_mean_std

# Save to CSV
print(formatted_stats)
basepath = os.path.dirname(csv_file)
ofile = os.path.join(basepath,'mean_std.csv')
formatted_stats.to_csv(ofile)
print(f'Mean and std deviation suscepibily values saved at: {ofile}')
