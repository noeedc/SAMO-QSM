"""
File: normalize.py
Description: This script provides functions for normalizing the pixel intensities of NIfTI images.
Author: Noée Ducros-Chabot

Dependencies:
- numpy
- nibabel
- glob
- os

Functions:
1. normalize(img, new_min=-1.0, new_max=1): Normalize an input NIfTI image to the range [-1, 1].
2. normalize_pi(img): Normalize an input NIfTI image to the range [-pi, pi].
3. apply_norm_pi(input_file, output_suffix='_pi', save=True): Normalize pixel intensities and save the result.
4. normalize_nifti(input_file, output_file, new_min, new_max): Normalize pixel intensities and save to a specified file.

Usage:
1. Each function serves a specific purpose and can be used independently.
2. Update the script's main block to demonstrate the usage of these functions with your specific use case.
"""

import numpy as np
import nibabel as nib 
import glob 
import os

def normalize(img, new_min =-1.0, new_max = 1):
    """
    Normalize an input NIfTI image to the range [-1, 1].

    Parameters
    ----------
    img : numpy.ndarray
        The input 3D or 4D NIfTI image data as a NumPy array.

    Returns
    -------
    numpy.ndarray
        The normalized image data as a NumPy array with the same shape as the input.
    """
    # Compute the minimum and maximum pixel values in the input image.
    img_min = img.min()
    img_max = img.max()

    # Perform the normalization operation on the input image.
    norm_img = (img - img_min) * (new_max - new_min) / (img_max - img_min) + new_min

    return norm_img

def normalize_pi(img):
    """
    Normalize an input NIfTI image to the range [-pi, pi].

    Parameters
    ----------
    img : numpy.ndarray
        The input 3D or 4D NIfTI image data as a NumPy array.

    Returns
    -------
    numpy.ndarray
        The normalized image data as a NumPy array with the same shape as the input.
    """
    # Compute the minimum and maximum pixel values in the input image.
    img_min = -4096
    img_max = 4096
    # img_min = img.min()
    # img_max = img.max()

    # Define the new minimum and maximum values to which the input image will be scaled.
    new_min = -np.pi
    new_max = np.pi

    # Perform the normalization operation on the input image.
    norm_img = (img - img_min) * (new_max - new_min) / (img_max - img_min) + new_min

    return norm_img


def apply_norm_pi(input_file, output_suffix='_pi', save = True):
    # TO DO : Change name to apply norm pi 
    """
    Normalize the pixel intensities of a NIfTI image and save the result.

    Parameters
    ----------
    input_file : str
        Path to the input NIfTI image file to normalize.
    output_suffix : str, optional
        Suffix to append to the input file name to generate the output file name.
        Default is 'phs_pi_norm'.

    Returns
    -------
    str
        Path to the output NIfTI image file that was saved.
    """
    # Load the input image data from the specified file.
    img_nii = nib.load(input_file)
    img = img_nii.get_fdata()

    # Normalize the image data.
    norm_img = normalize_pi(img)

    if save:
        # Create a new NIfTI image object with the normalized data and the same header as the input.
        phs_nii = nib.nifti1.Nifti1Image(norm_img, None, img_nii.header)

        # Generate the output file name by replacing 'phs' in the input file name with the specified suffix.
        output_file = input_file.replace('.nii', output_suffix+'.nii')

        # Save the normalized image data to the specified output file.
        nib.save(phs_nii, output_file)

    return norm_img


def normalize_nifti(input_file, output_file, new_min, new_max):
    """
    Normalize pixel intensities of a NIfTI image and save the result to a new file.

    Parameters
    ----------
    input_file : str
        Path to the input NIfTI image file to be normalized.
    output_file : str
        Path to save the normalized NIfTI image.
    new_min : float
        The minimum value to which the pixel intensities will be scaled.
    new_max : float
        The maximum value to which the pixel intensities will be scaled.

    Returns
    -------
    None

    Notes
    -----
    - The function loads the input NIfTI image from the specified file path using nibabel.
    - It normalizes the pixel intensities of the image using the `normalize` function.
    - The normalized image is saved to the specified output file.
    - The output file has the same header information as the input file.

    Example
    -------
    >>> input_path = '/path/to/input/image.nii'
    >>> output_path = '/path/to/output/normalized_image.nii'
    >>> min_value = -1.0
    >>> max_value = 1.0
    >>> normalize_nifti(input_path, output_path, min_value, max_value)
    """
    # Load the input image data from the specified file.
    img_nii = nib.load(input_file)
    img = img_nii.get_fdata()

    # Normalize the image data.
    norm_img = normalize(img, new_min=new_min, new_max=new_max)

    # Create a new NIfTI image object with the normalized data and the same header as the input.
    norm_nii = nib.nifti1.Nifti1Image(norm_img, None, img_nii.header)

    # Save the normalized image data to the specified output file.
    nib.save(norm_nii, output_file)

if __name__ == "__main__":
    # Example usage 1
    basefolder = '/home/magic-chusj-2/Documents/E2022/AMONI/Phantom2'
    files = glob.glob(os.path.join(basefolder, '*/*PHASE.nii'))
    files = sorted(files)
    for file in files: 
        output_file = apply_norm_pi(file)
        print(f'Created: {output_file}')

    # Example usage 2
    # Normalize a single input file.
    # apply_norm('/home/magic-chusj-2/Documents/E2022/AMONI/Phantom2/Test1_Normal_Orientation/High_Res_PHASE.nii')
