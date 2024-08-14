"""
File: extract_echo_times.py
Description: This script extracts echo times (TE) from a list of JSON files.
Author: Noée Ducros-Chabot

Dependencies:
- os
- glob
- json

Usage:
1. Specify the path where the JSON files containing echo time information are located.
2. The script finds JSON files matching the pattern '*_e?.json' in the specified directory.
3. Echo times are extracted from the JSON files and printed.

Note: Update the 'path' variable with the correct directory path before running the script.
"""

import os 
import glob
import json

def extract_echo_times(jsn_files):
    """
    Extract echo times (TE) from a list of JSON files.

    Parameters:
        jsn_files (list of str): List of file paths to JSON files containing TE information.

    Returns:
        list of float: List of extracted echo times.
    """
    echo_times = []

    for jsn_file in jsn_files:
        # Load the JSON data from the file
        with open(jsn_file, 'r') as json_file:
            data = json.load(json_file)

        # Extract the echo time (TE) from the JSON data
        te = data['EchoTime']

        echo_times.append(te)

    return echo_times

if __name__ == '__main__':
    # Specify the path where the JSON files are located
    path = '/home/magic-chusj-2/Documents/E2022/downsampled_dataset/2.4mm/central'

    # Find JSON files that match the pattern '*_e?.json' in the specified directory
    TE_files = sorted(glob.glob(os.path.join(path, '*_e?.json')))

    # Extract the echo times from the JSON files
    TEs = extract_echo_times(TE_files)

    # Print the extracted echo times
    print(TEs)
