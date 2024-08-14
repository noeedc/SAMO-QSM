"""
Script: Image Quality Metrics Calculation and Aggregation

Description:
This script calculates image quality metrics, including 1-NRMSE, XSIM, PSNR, and NMI,
for a set of images compared to a target image. It organizes the results into a structured DataFrame.

Author: Noée Ducros-Chabot
Date: Decembre 14th 2023

Dependencies:
- numpy
- os
- skimage.metrics (normalized_root_mse, peak_signal_noise_ratio, normalized_mutual_information)
- nibabel
- pandas
- glob
- re

Usage:
1. Ensure the required libraries are installed: numpy, os, scikit-image, nibabel, pandas.
   You can install them using: pip install numpy scikit-image nibabel pandas

2. Update the 'target_file', 'data_folder', 'msk_file' and 'resolutions' variables according to your data.

3. Run the script to calculate image quality metrics and save the results as a CSV file.

4. The resulting DataFrame is saved as a CSV file named 'image_quality_metrics.csv' in the specified 'data_folder/results' directory.

Note: This script assumes a specific structure of input files and labels and requires the scikit-image and nibabel libraries.

"""
import numpy as np
import os
from skimage.metrics import normalized_root_mse, peak_signal_noise_ratio, normalized_mutual_information
import nibabel as nib
import pandas as pd
import glob
import re 

def extract_resolution(input_string):
    """
    Extract the resolution substring (e.g., '1.9mm', '1.05mm') from the given input string.

    Parameters:
        input_string (str): The input string.

    Returns:
        str: The matched resolution substring or an empty string if no match is found.
    """
    # Define the pattern
    pattern = re.compile(r'(\d+\.\d+mm)')

    # Use findall to get all matches in the string
    matches = pattern.findall(input_string)

    # Return the result
    return matches[0] if matches else ""

def addToDF(df, **kwargs):
    """
    Add key-value pairs as new columns to a pandas DataFrame.

    This function takes a pandas DataFrame and a set of keyword arguments (key-value pairs),
    and adds each key as a new column to the DataFrame with the corresponding values.

    Parameters:
    df (pandas.DataFrame): The DataFrame to which new columns will be added.
    **kwargs: Keyword arguments where the key becomes the column name and the value is added to the column.

    Returns:
    pandas.DataFrame: The DataFrame with the new columns added.
    """
    for key, value in kwargs.items():
        df[key] = value
    return df 

def xsim(img1, img2, k1=0.01, k2=0.001, L=1):
    """
    Compute the XSIM (Susceptibility-Adapted Structural Similarity Index) between two images.

    Parameters:
    img1 (numpy.ndarray): First image.
    img2 (numpy.ndarray): Second image.
    k1 (float): Stability constant for mean calculations. Default is 0.01.
    k2 (float): Stability constant for variance calculations. Default is 0.001.
    L (float): Dynamic range of pixel values (e.g., 1 for normalized images). Default is 1.

    Returns:
    float: XSIM value between the two images.
    """
    # Set C1 and C2 constants
    C1 = (k1*L)**2
    C2 = (k2*L)**2

    # Convert the images to float
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Calculate means
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    # Calculate variances
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.cov(img1.flatten(), img2.flatten())[0][1]

    # Calculate SSIM
    numerator = (2*mu1*mu2 + C1) * (2*sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    xsim_val = numerator / denominator

    return xsim_val

def calculate_metrics(files, source_img, msk):
    """
    Calculate image quality metrics for a set of images.

    Parameters:
    files (list): List of file paths for images to be evaluated.
    source_img (numpy.ndarray): Reference image for comparison.
    msk (numpy.ndarray): Mask indicating the region of interest.

    Returns:
    pandas.DataFrame: DataFrame containing calculated metrics for each image.
    """
    NRMSEs = []
    XSIMs = []
    PSNRs = []
    NMIs = []

    for file in files:
        img_nii = nib.load(file)
        img = img_nii.get_fdata()
        img[~msk] = 0

        nrmse = abs(1-normalized_root_mse(source_img, img)) # Don't really get why values ouput are not between 0 and 1
        NRMSEs.append(nrmse)

        xsim_value = xsim(source_img, img)
        XSIMs.append(xsim_value)

        PSNR = peak_signal_noise_ratio(source_img, img, data_range=img.max() - img.min())
        PSNRs.append(PSNR)

        NMI = normalized_mutual_information(source_img, img)
        NMIs.append(NMI)

    if 'gz' not in files[0]:  # assumes all images have the same suffix
        names = [os.path.basename(file).replace('.nii', '') for file in files]
    else:
        names = [os.path.basename(file).replace('.nii.gz', '') for file in files]

    df = pd.DataFrame({'Orientation': names, '1-NRMSE': NRMSEs, 'XSIM': XSIMs, 'PSNR': PSNRs, 'NMI': NMIs})
    return df

if __name__ == "__main__":
    # Script to calculate image quality metrics and save as csv file for data path.

    # Target QSM map : 5 orientations COSMOS
    target_file = '/home/magic-chusj-2/Documents/E2022/DATA/ex_ds_dataset/data/1.05mm/results/HR_localfields/w0.0_tkd0.00/cosmos_ori5_CEFLR.nii'
    
    data_folder = '/home/magic-chusj-2/Documents/E2022/DATA/ex_ds_dataset/data/1.05mm' # Can be changed to *mm if multi-resolution was reconstructed
    if '*' in data_folder : 
        msk_file = sorted(glob.glob(os.path.join(data_folder, 'central/*bckremoval_msk.nii')))[0] #Assumes smallest res is first after sorting
    msk_file = glob.glob(os.path.join(data_folder, 'central/*bckremoval_msk.nii'))[0]
    resolutions = ['1.05mm', '1.9mm'] # Input high and low resolution values (even if simulated). Make sure it is coherent with the name of folder if resolution is identified in path.

    # MULTI-ORIENTATION DATA
    img_files = sorted(glob.glob(os.path.join(data_folder, 'results/*_localfields/w*_tkd*/*osmos*.nii')))

    # Find number of orientations used for each file
    nOri = [int(num) for x in img_files for num in re.findall(r'\d+', os.path.basename(x))]

    # Find parameter for recon of each file
    w = [float(num) for x in img_files for num in re.findall(r'w(\d+\.\d+)_tkd', os.path.basename(os.path.dirname(x)))]
    tkd = [float(num) for x in img_files for num in re.findall(r'tkd(\d+\.\d+)', os.path.basename(os.path.dirname(x)))]
    inversion = ['COSMOS' if 'w0.0_tkd0.00' in file else 'mCOSMOS' for file in img_files]

    # Find local field types
    methods = ['ILR' if 'ILR' in file else 'pred' if 'pred' in file else 'HR' for file in img_files]
    
    # Verifie si la resolution est écrite dans les path et si elle n'égale pas la haute résolution.
    if all('mm' in file for file in img_files) and not all(resolutions[0] in file for file in img_files):
        res = [extract_resolution(file) for file in img_files]
    else : 
        res = [resolutions[0] if 'HR' in file else f's{resolutions[1]}' for file in img_files]

    # Find orientations used for reconstruction
    orientation = [os.path.basename(x).replace('.nii', '') for x in img_files]
    combinations = [os.path.basename(x).replace('.nii', '').split('_')[-1] for x in img_files]

    msk_nii = nib.load(msk_file)
    msk = msk_nii.get_fdata().astype(bool)

    target_nii = nib.load(target_file)
    target_img = target_nii.get_fdata()
    target_img[~msk] = 0

    # SINGLE ORIENTATION DATA
    singleOri_subfolders  = sorted(glob.glob(os.path.join(data_folder, 'results/singleOri')))
    if singleOri_subfolders:
        single_qsm_files = sorted(glob.glob(os.path.join(data_folder, 'results/singleOri/*/*Chimap.nii*')))
        
        for file in single_qsm_files:
            basename = os.path.basename(file).replace('.nii','').split('_')
            folder_path = os.path.dirname(file).split('/')

            orientation.append(folder_path[-1])
            inversion.append(basename[0])
            methods.append('HR')
            nOri.append(1)
            res.append(resolutions[0]) # Assumes single orientation QSM are reconstructed from high resolution data

            combinations.append('')
            tkd.append('')
            w.append('')

        img_files.extend(single_qsm_files)

    df = calculate_metrics(img_files, target_img, msk)
    df = addToDF(df, nOri = nOri, Method = methods, Combination = combinations, w = w, tkd = tkd, Inversion = inversion, Res = res, Orientation = orientation)

    # Reorder columns 
    df = df[['nOri', 'Method', 'Inversion', 'Combination', 'Orientation', 'Res', 'w', 'tkd', '1-NRMSE', 'XSIM', 'PSNR', 'NMI']]

    # Save df as csv file 
    if '*' in data_folder:
        basefolder = os.path.dirname(data_folder.split('*')[0])
        ofile = os.path.join(basefolder, 'image_quality_metrics.csv')
    else :
        ofile = os.path.join(data_folder, 'results/image_quality_metrics.csv')
    df.to_csv(ofile)
    print(f'Similarity Metric CSV file created: {ofile}')
