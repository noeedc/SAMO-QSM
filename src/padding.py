"""
File: padding.py
Author: Noée Ducros-Chabot
Description:
    This script provides functions for padding and removing padding from a 3D numpy array.
    The primary purpose is to ensure even dimensions for each slice along each axis.

Functions:
    - pad_to_even_slices_3d(arr):Pad a 3D numpy array to ensure even dimensions along each axis.
    - remove_padding_3d(padded_arr, padding):Remove padding from a previously padded 3D numpy array.

Usage:
    The script can be run independently to demonstrate the usage of the provided functions.
    It loads a sample 3D numpy array from a NIfTI file, pads it, performs operations (if needed),
    and then removes the padding to restore the original shape.
"""

import numpy as np
import nibabel as nib

def pad_to_even_slices_3d(arr):
    """
    Pad a 3D numpy array to ensure even dimensions along each axis.

    Parameters:
    - arr (numpy.ndarray): The input 3D numpy array.

    Returns:
    - numpy.ndarray: The padded 3D numpy array.
    - list of tuples: The padding applied to each dimension.

    Notes:
    - The function checks each dimension of the input array and pads if needed to make it even.
    """

    # Get the shape of the input array
    shape = arr.shape
    
    # Calculate the padding needed for each dimension to make them even
    padding = [(0, 0)] * 3  # Assuming a 3D array
    
    for i in range(3):
        if shape[i] % 2 != 0:
            padding[i] = (0, 1)
    
    # Pad the 3D array with zeros to make dimensions even
    padded_arr = np.pad(arr, padding, mode='constant')
    
    return padded_arr, padding

def remove_padding_3d(padded_arr, padding):
    """
    Remove padding from a previously padded 3D numpy array.

    Parameters:
    - padded_arr (numpy.ndarray): The 3D numpy array with padding.
    - padding (list of tuples): The padding applied to each dimension.

    Returns:
    - numpy.ndarray: The 3D numpy array with padding removed.

    Notes:
    - The function restores the original shape by removing the padding applied during the padding process.
    """
    # Remove the padding from the padded array
    original_shape = tuple(padded_arr.shape[i] - sum(p) for i, p in enumerate(padding))
    original_arr = padded_arr[:original_shape[0], :original_shape[1], :original_shape[2]]
    
    return original_arr

if __name__ == "__main__":
    # Example usage:
    file = '/home/magic-chusj-2/Documents/E2022/CNN-Data(copy)/sub-010036/ses-01/anat/sub-010036_ses-01_acq-phase_GRE_cropped_1mm_norm.nii.gz'
    input_array_3d = nib.load(file).get_fdata()  # Replace with your own 3D numpy array

    # Pad the array to ensure even dimensions
    padded_array_3d, padding = pad_to_even_slices_3d(input_array_3d)

    # Perform some operations on the padded array if needed

    # Revert back to the original shape
    original_array_3d = remove_padding_3d(padded_array_3d, padding)

    print("Original Shape:", input_array_3d.shape)
    print("Padded Shape:", padded_array_3d.shape)
    print("Restored Shape:", original_array_3d.shape)

    # Now 'original_array_3d' should match the original shape.
