"""
MRI Data Preparation Script
---------------------------

This script prepares MRI data by merging separate echoes into 4D NIfTI files. It assumes that magnitude files
have the suffix '*e?.nii' and phase files have the suffix '*e?_ph.nii'. The processed magnitude and phase images
are saved in the same folder with the '_magnitude.nii' and '_phase.nii' suffixes, respectively.

Author: Noée Ducros-Chabot
Date: 23/11/2023

Usage: Modify the 'data_folder' variable with the path to the folder containing MRI data files.
       Ensure that the required dependencies are installed.

"""

import nibabel as nib
import os
import glob
from src.nifti_file_tools import joinTo4D
from src.file_manager import get_subfolders

def prepare_data(data_path):
    """
    Prepares MRI data from a specified path by merging seperate echos into 4D nifti files.

    Parameters:
    - data_path (str): The path to the folder containing MRI data files.

    Returns:
    - None: The function saves processed magnitude and phase images to the same folder.

    Notes:
    - Assumes that magnitude files are have suffix '*e?.nii' and phase files are suffix '*e?_ph.nii'.
    - The function joins multiple 3D volumes into a single 4D NIfTI volume for both magnitude and phase images.
    - Output files are saved with '_magnitude.nii' and '_phase.nii' suffixes.

    Example:
    >>> data_path = "/path/to/mri_data_folder"
    >>> prepare_data(data_path)
    """
    mag_files = sorted(glob.glob(os.path.join(data_path, '*e?.nii')))
    phs_files = sorted(glob.glob(os.path.join(data_path, '*e?_ph.nii')))

    # Extract file basename
    basename_image = os.path.basename(mag_files[0]).split('_')
    basename_image = "_".join(basename_image[:-2])

    # Create output file names
    suffix_image = '.nii'
    fname_out_magnitude = os.path.join(data_path, basename_image + '_magnitude' + suffix_image)
    fname_out_phase = os.path.join(data_path, basename_image + '_phase' + suffix_image)

    magnitude, _, hdr = joinTo4D(mag_files)
    phase, _, _ = joinTo4D(phs_files)

    # Saving the magnitude and phase images
    image_magnitude = nib.nifti1.Nifti1Image(magnitude.copy(), None, hdr)
    nib.save(image_magnitude, fname_out_magnitude)
    image_phase = nib.nifti1.Nifti1Image(phase.copy(), None, hdr)
    nib.save(image_phase, fname_out_phase)


if __name__ == "__main__":
    # Example usage 
    data_folder = '/home/magic-chusj-2/Documents/E2022/DATA/ex_ds_dataset/data/1.05'
    subfolders = get_subfolders(data_folder)
    for subfolder in subfolders:
        prepare_data(subfolder)
