"""
Script: Calculate Susceptibility Values for Regions of Interests (ROI)

Description:
This script extracts mean and standard deviation susceptibility values from multi-orientation data and aggregates the results.
It utilizes NIfTI image processing with the nibabel library and organizes the data into a structured DataFrame.

Author: Noée Ducros-Chabot
Date: Decembre 14th 2023

Dependencies:
- numpy
- nibabel
- pandas
- glob
- os
- re
- eval.calculate_metrics.addToDF (imported from a separate file)

Usage:
1. Ensure the required libraries are installed: numpy, nibabel, pandas.
   You can install them using: pip install numpy nibabel pandas

2. Ensure the 'eval.calculate_metrics.addToDF' function is defined in a separate file and is imported successfully.

3. Update the 'basefolder', 'msk_file', 'labelmap_file', 'resolutions', 'folders' and 'suffixs' variables.

4. Run the script to extract and aggregate susceptibility values from the specified multi-orientation data.

5. The resulting DataFrame is saved as a CSV file named 'whole_mean_susceptibilities.csv' in the specified 'basefolder'.

Note: This script assumes a specific structure of input files and labels, and it uses the 'eval.calculate_metrics.addToDF' function.

"""
import numpy as np
import nibabel as nib
import numpy as np
import glob
import pandas as pd
import os
import re

from eval.calculate_metrics import addToDF, extract_resolution

def extract_avg_std_with_labels(qsm_file, msk_file, labelmap_file, labels_csv):
    qsm_nii = nib.load(qsm_file)
    qsm = qsm_nii.get_fdata()

    msk_nii = nib.load(msk_file)
    msk = msk_nii.get_fdata()

    labelmap_nii = nib.load(labelmap_file)
    labelmap = labelmap_nii.get_fdata()
    labelmap = labelmap * msk

    struct_label = pd.read_csv(labels_csv)

    averages = []
    stds = []
    data = []
    for value in struct_label['Value']:
        avg = np.mean(qsm[labelmap == value])
        std = np.std(qsm[labelmap == value])
        pixel_data = qsm[labelmap == value]

        averages.append(avg)
        stds.append(std)
        data.append(pixel_data)

    struct_label['average'] = averages
    struct_label['std'] = stds

    return struct_label, data


# Script to extract mean and std susceptibility values from multi-orientation data
if __name__ == "__main__":
    # TO BE CONTINUED  : test with multi-resolution results 
    basefolder = "/home/magic-chusj-2/Documents/E2022/DATA/ex_ds_dataset/data/*mm/results"

    msk_file = '/home/magic-chusj-2/Documents/E2022/DATA/downsampled_dataset_nCNN/1.9mm/central/dicom_gre_IPAT2_pPF_sPF_3D_6echoes_1.05_20230829132927_3_bckremoval_msk.nii'
    labelmap_file = '/home/magic-chusj-2/Documents/E2022/DATA/downsampled_dataset/1.05mm/central/reg_atlas/reg_MuSus-100_Atlas_labelmap_edited.nii'
    
    folders = ["ILR_localfields/w0.5_tkd0.20", "pred_localfields/w0.5_tkd0.20", 'HR_localfields/w0.0_tkd0.00', 'singleOri/*'] # Just the folders you are interested in
    suffixs = ["*osmos_ori3*.nii*","*osmos_ori4*.nii*", "*osmos_ori5*.nii*", "*Chimap.nii*" ]

    resolutions = ['1.05mm', '1.9mm'] # [High res, simulated res], code needs to be changes if more than one resolution simulated

    labels_csv = './config/labelmap.csv'

    qsm_files = [sorted(glob.glob(os.path.join(basefolder, folder, suffix))) for suffix in suffixs for folder in folders]
    qsm_files = sorted([item for sublist in qsm_files for item in sublist if 'diff' not in item]) # Makes sure that it doesn't take difference images

    dfs = []
    for qsm_file in qsm_files :
        struct_label, _ = extract_avg_std_with_labels(qsm_file, msk_file, labelmap_file, labels_csv)
        
        n_ROI = len(struct_label)

        basename = os.path.basename(qsm_file).replace('.nii','').split('_')
        folder_path = os.path.dirname(qsm_file).split('/')
        if 'singleOri' in qsm_file:
            orientation = folder_path[-1]
            inversion = basename[0]
            method = inversion
            nOri = 1
            resolution = ['1.05mm'] *n_ROI

        else :
            orientation = basename[-1]
            nOri = int(re.findall(r'\d+', os.path.basename(qsm_file))[0]) 
            
            if 'mm' in qsm_file and resolutions[0] not in qsm_file :
                resolution = [extract_resolution(qsm_file)] *n_ROI
            else:
                resolution = [f's{resolutions[1]}']*n_ROI # If other than high resolution not part of the path, assumes the data is simulated.
        
            if folder_path[-2] == 'pred_localfields':
                inversion = 'SAMO-QSM'
            
            elif folder_path[-2] == 'ILR_localfields':
                inversion = 'ILR-mCOSMOS'
            
            elif folder_path[-2] == 'HR_localfields':
                inversion = 'COSMOS'
                resolution = ['1.05mm'] *n_ROI
            method = f'{nOri} ori. {inversion}'

        struct_label = addToDF(struct_label, nOri = nOri, Res = resolution, Inversion = [inversion]*n_ROI, Orientation = [orientation]*n_ROI, Method = method)
        dfs.append(struct_label)
        
    df = pd.concat(dfs).reset_index(drop = True)

    # Save df
    if '*' not in basefolder:
        ofile = os.path.join(basefolder, 'whole_mean_susceptibilities.csv')
    else : 
        basefolder = basefolder.split('*')[0]
        ofile = os.path.join(basefolder, 'whole_mean_susceptibilities.csv')
    df.to_csv(ofile)
    print(f'Susceptibility CSV file created: {ofile}')
