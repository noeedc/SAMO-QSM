"""
Script: Statistical Analysis of Image Quality Metrics

Description:
This script performs statistical analysis on image quality metrics obtained from different reconstruction methods.
The analysis includes paired or Welch's t-tests with Bonferroni correction to compare the performance of SAMO-QSM
against other reconstruction methods.

Author: Noée Ducros-Chabot
Date: December 14th 2023

Dependencies:
- numpy
- scipy.stats
- pandas
- os
- warnings

Usage:
1. Ensure the required libraries are installed: numpy, scipy, pandas.
   You can install them using: pip install numpy scipy pandas

2. Update the 'results_csv' variable with the path to the image quality metrics CSV file. 

3. Modify the query_conditions depending on what method you want to calculate statistical significance for.

4. Run the script to perform statistical tests and save the results in a CSV file named 'stat_significance.csv'.

5. The significance results include t-statistic, p-value, and whether the difference is significant after Bonferroni correction.

Note: This script assumes a specific structure of the input DataFrame and requires the specified dependencies.

"""
from numpy.core.numeric import NaN
from scipy import stats
import pandas as pd
import os 
import warnings


def check_normality(data, alpha=0.05):
    """
    Check the normality of a dataset using the Shapiro-Wilk test and visualizations.

    Parameters:
        data (numpy.ndarray): The dataset to check for normality.
        alpha (float): Desired significance level for the Shapiro-Wilk test.

    Returns:
        is_normal (bool): True if the data is normally distributed, False otherwise.
    """
    statistic, p_value = stats.shapiro(data)
    is_normal = p_value > alpha

    return is_normal

def ttest_with_bonferroni_correction(data1, data2, alpha):
    """
    Perform a paired or Welch's t-test with Bonferroni correction for comparing two sets of data.

    Parameters:
        data1 (numpy.ndarray): First set of data.
        data2 (numpy.ndarray): Second set of data.
        alpha (float): Desired significance level.

    Returns:
        t_statistic (float): Calculated t-statistic.
        p_value (float): Calculated p-value.
        is_significant (bool): True if the result is significant after Bonferroni correction.
    """
    num_comparisons = len(data1) * len(data2)
    adjusted_alpha = alpha / num_comparisons

    if len(data1) == len(data2):
        t_statistic, p_value = stats.ttest_rel(data1, data2) # When sample size is the same, a paired t-test is necessary 
        # since we are using the same images for reconstruction i.e. CLR, CEF, ect.
    else:
        t_statistic, p_value = stats.ttest_ind(data1, data2, equal_var=False)

    is_significant = p_value < adjusted_alpha

    return t_statistic, p_value, is_significant

def perform_statistical_tests(df, SAMO_df, method, methods_list, metrics_list, t_stats_list, p_values_list, significances_list, alpha=0.05):
    for column in df.columns[1:]:
        methods_list.append(method)
        metrics_list.append(column)

        singleOri_values = df[column]
        SAMO_values = SAMO_df[column]
        diff = SAMO_values - singleOri_values
        if not (check_normality(diff)): 
           warnings.warn(f"The data for {method}'s {column} is not normally distributed")
           t_statistic, p_value, is_significant = NaN, NaN, NaN
        else :
            t_statistic, p_value, is_significant = ttest_with_bonferroni_correction(singleOri_values, SAMO_values, alpha=alpha)

        t_stats_list.append(t_statistic)
        p_values_list.append(p_value)
        significances_list.append(is_significant)
    return df


if __name__ == "__main__":
    # Load the joined DataFrame from the CSV file
    results_csv = '/home/magic-chusj-2/Documents/E2022/DATA/ex_ds_dataset/data/1.05mm/results/image_quality_metrics.csv'
    joinedDF = pd.read_csv(results_csv, index_col=0).reset_index(drop = True)
    
    results_folder = os.path.dirname(results_csv)

    # Define the columns of interest
    desired_columns = ['Inversion', '1-NRMSE', 'XSIM', 'PSNR', 'NMI']
    SAMO_query  = '((tkd == 0.20) & (w == 0.5)) & (Method == "pred") & (nOri == 3)'
    SAMO_df = joinedDF.query(SAMO_query)[desired_columns]
    
    # If csv file contain data from single orientation methods
    if any(joinedDF['nOri'] == 1):
        singleOri_df = joinedDF.query('nOri == 1')[desired_columns]

    
    ILR_query = '((tkd == 0.20) & (w == 0.5)) & (Method == "ILR") & (nOri == 3)'
    ILR_df = joinedDF.query(ILR_query)[desired_columns]

    cosmos3_query = '(Method == "HR") & (nOri == 3)'
    cosmos3_df = joinedDF.query(cosmos3_query)[desired_columns]

    # Loop through
    methods = [] 
    metrics = []
    t_stats = []
    p_values = []
    significances = []
    
    df = perform_statistical_tests(ILR_df, SAMO_df, method='ILR-mCOSMOS', methods_list=methods, metrics_list=metrics, t_stats_list=t_stats, p_values_list=p_values, significances_list=significances, alpha=0.05)
    perform_statistical_tests(cosmos3_df, SAMO_df, method='COSMOS-3ori', methods_list=methods, metrics_list=metrics, t_stats_list=t_stats, p_values_list=p_values, significances_list=significances, alpha=0.05)
    
    #Single Orientation data
    if any(joinedDF['nOri'] == 1):
        for method in singleOri_df['Inversion'].unique():
            single_df = singleOri_df.query(f'Inversion == "{method}"')
            perform_statistical_tests(single_df, SAMO_df, method=method, methods_list=methods, metrics_list=metrics, t_stats_list=t_stats, p_values_list=p_values, significances_list=significances, alpha=0.05)
    
    # Create a DataFrame for statistical significance results
    sign_df = pd.DataFrame({"Method": methods, "Metric": metrics, 'T-stat': t_stats, 'P-value': p_values, 'Significant': significances })
    
    # Save the statistical significance DataFrame as a CSV file 
    ofile = os.path.join(results_folder, 'stat_significance.csv')
    sign_df.to_csv(ofile)
    print(f'Statistical significance saved at: {ofile}')