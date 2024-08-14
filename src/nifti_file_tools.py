import nibabel as nib 
import numpy as np 
import os

"""
File: nifti_file_tools.py

Description:
This file contains utility functions for processing NIfTI files, specifically focusing on operations related to 3D and 4D volumes. 
Functions include combining multiple 3D volumes into a single 4D NIfTI volume, aggregating and saving 4D NIfTI files, creating repeated NIfTI images, 
and merging matrices from text files into a 4D NIfTI image. These functions leverage the capabilities of the nibabel and numpy libraries for neuroimaging 
data manipulation.

Author: Noée Ducros-Chabot
Date: 22/11/2023
"""
def get_voxel_size(nifti_file):
    """
    Get the voxel size from a NIfTI file.

    Parameters:
        nifti_file (str): File path to the NIfTI file.

    Returns:
        tuple: Tuple containing the voxel sizes in the X, Y, and Z dimensions.
    """
    header = nib.load(nifti_file).header
    voxel_size = header.get_zooms()[:3]  # Extract voxel size for the first three dimensions (X, Y, Z)
    return list(voxel_size)

def joinTo4D(files):
    """
    Combines 3D volumes from multiple NIfTI files into a single 4D NIfTI volume.

    Parameters:
    - files (list): A list of file paths to NIfTI (.nii) files containing 3D volumes.

    Returns:
    - data_4d (numpy.ndarray): A 4D array containing concatenated 3D volumes from input files.
    - img_nii (nibabel.nifti1.Nifti1Image): The NIfTI image object representing the 4D data.
    - hdr (nibabel.nifti1.Nifti1Header): The NIfTI header object associated with the 4D data.

    Notes:
    - If any of the input 3D volumes has more than one volume (e.g., multiple echoes), 
      only the first volume is considered for concatenation.
    - The datatype of the resulting NIfTI image is set to float (float32), and the bit depth is 32.
    """
    # Initialize an empty list to store individual 3D volumes
    imgs = []

    # Loop through each file path in the input list
    for file in files:
        # Load the NIfTI image from the file
        img_nii = nib.load(file)

        # Extract the 3D volume data from the NIfTI image
        img = img_nii.get_fdata()

        # If the 3D volume has more than one volume, consider only the first volume
        if len(img.shape) > 3:
            img = img[:, :, :, 0]

        # Append the 3D volume to the list
        imgs.append(img)

    # Create a copy of the header from the last loaded NIfTI image
    hdr = img_nii.header.copy()

    # Update the 4th dimension of the header to reflect the total number of input files
    hdr['dim'][4] = len(files)

    # Set the datatype of the resulting NIfTI image to float (float32)
    hdr['datatype'] = 16

    # Set the bit depth of the resulting NIfTI image to 32
    hdr['bitpix'] = 32

    # Concatenate the 3D volumes into a 4D array along the last axis
    data_4d = np.stack(imgs, axis=-1).astype(np.float32)

    # Return the concatenated 4D data, NIfTI image object, and NIfTI header object
    return data_4d, img_nii, hdr

def aggregate_and_save_4D_nifti(img_files, fout, overwrite= True):
    """
    Aggregate and save 4D NIfTI files.

    Parameters:
        img_files (list): List of 3D NIfTI file paths to be aggregated.
        fout (str): File path for the output 4D NIfTI file.
        overwrite (bool): If True, overwrite the existing file; if False, provide a warning if the file exists.

    Returns:
        np.ndarray: 4D array containing the aggregated data.
    """
    if os.path.exists(fout) and not overwrite:
        response = input(f"Warning: File '{fout}' already exists. Do you want to overwrite it? (y/n): ")
        if response.lower() != 'y':
            print("Aborting operation.")
            return
    
    img4D, img_nii, img_hdr = joinTo4D(img_files)
    mag4D_nii = nib.Nifti1Image(img4D.copy(), img_nii.affine, img_hdr)
    if os.path.exists(fout) and overwrite:
        print(f"4D NIfTI file '{fout}' already existed and has been overwritten with the new data.")
    else:
        print(f"4D NIfTI file saved at '{fout}'")
    nib.save(mag4D_nii, fout)
    
    return img4D

def create_repeated_nifti(input_file, repetitions, output_file):
    """
    Creates a 4D NIfTI image by repeating the 3D data from an input NIfTI image.

    Parameters:
    - input_file (str): The file path to the input NIfTI (.nii) file.
    - repetitions (int): The number of times to repeat the 3D data to create the 4D array.
    - output_file (str): The file path to save the newly created 4D NIfTI image.

    Returns:
    - repeated_data (numpy.ndarray): The 4D array containing the repeated 3D data.

    Notes:
    - The function loads the input NIfTI image from the specified file path using nibabel.
    - It extracts the voxel data as a NumPy array and repeats the 3D data to create a 4D array.
    - A new NIfTI image is created from the repeated data and the original affine transformation.
    - The newly created 4D image is saved to the specified output file.
    - The function returns the 4D array containing the repeated 3D data.
    """
    # Load the input NIfTI image from the specified file path
    nifti_img = nib.load(input_file)
    
    # Extract the voxel data from the input NIfTI image as a NumPy array
    data = nifti_img.get_fdata()

    # Repeat the 3D data to create a 4D array
    repeated_data = np.repeat(data[..., np.newaxis], repetitions, axis=-1)

    # Create a new NIfTI image with the repeated data and the original affine transformation
    new_nifti_img = nib.Nifti1Image(repeated_data, affine=nifti_img.affine)

    # Save the new NIfTI image to the specified output file
    nib.save(new_nifti_img, output_file)

    # Return the 4D array containing the repeated 3D data
    return repeated_data


def merge_matrices_to_4Dnifti(input_files):
    """
    Merge 4x4 matrices from text files into a 4D 3x3 Nifti image.
    """
    # Create empty 4D array to store merged matrices
    merged = np.empty((3, 3, len(input_files)))

    # Load each file and append to the 4D array
    for i, file in enumerate(input_files):
        matrix = np.loadtxt(file)
        merged[:, :, i] = matrix[:3, :3]

    # Save the merged 4D array as a nifti file
    return nib.Nifti1Image(merged, np.eye(4)), merged