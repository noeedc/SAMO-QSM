"""
Patch Generation and Image Reconstruction Script

Author: Noée Ducros-Chabot
Created: 12/06/2023
Last Modified : 23/11/2023

Description:
This script defines functions to randomly extract 3D patches from an image, generate patches using sliding window, and reconstruct an image from overlapping patches. The script also includes an example usage section.

Libraries Used:
- src.normalize.normalize
- numpy
- nibabel

Functions:
- random_patches_3D_with_indices
- generate_3D_rand_patches
- sliding_window_3d_with_indices
- generate_3D_sliding_window_patches
- reconstruct_image_with_overlap

Usage:
- Modify the file paths and parameters in the example usage section according to your specific task.
- Execute the script using the Python interpreter.

Note:
- Ensure that the required libraries are installed before running the script.
- Adjust the script parameters and configurations as needed.
"""

from src.normalize import normalize
import numpy as np
import nibabel as nib

def random_patches_3D_with_indices(image, patch_size, n):
    """
    Randomly extract 3D patches from an image with their corresponding indices.

    Args:
        image (ndarray): 3D image array.
        patch_size (tuple): Size of the patches to be extracted.
        n (int): Number of patches to extract.

    Returns:
        tuple: A tuple containing the patch indices and the extracted patches.
    """
    image_shape = image.shape
    patch_shape = tuple(patch_size) + (image_shape[-1],)  # Add channel dimension if applicable

    patches = []
    indices = []
    count = 0

    while count < n:
        i = np.random.randint(0, image_shape[0] - patch_shape[0] + 1)
        j = np.random.randint(0, image_shape[1] - patch_shape[1] + 1)
        k = np.random.randint(0, image_shape[2] - patch_shape[2] + 1)

        patch_indices = (i, j, k)
        patch = image[i:i+patch_shape[0], j:j+patch_shape[1], k:k+patch_shape[2], ...]
        indices.append(patch_indices)
        patches.append(patch)
        count += 1

    return np.array(indices), np.array(patches)

def generate_3D_rand_patches(ILR_file, HR_file, target_file, patch_size, n_patch, norm = True):
    """
    Generate random patches from interpolated low-resolution (ILR) and high-resolution (HR) images, along with their ground truth patches.

    Args:
        ILR_file (str): File path of the low-resolution image.
        HR_file (str): File path of the high-resolution image.
        target_file (str): File path of the target image (ground truth).
        patch_size (tuple): Size of the patches to be generated.
        n_patch (int): Number of patches to generate.
        norm (bool) : Boolean if high resolution image (magnitude) should be normalize (between -1 and 1)

    Returns:
        tuple: A tuple containing the patch indices, ILR patches, HR patches, and ground truth patches.
    """
    ILR_nii = nib.load(ILR_file)
    ILR_data = ILR_nii.get_fdata()

    HR_nii = nib.load(HR_file)
    HR_data = HR_nii.get_fdata()
    
    # If not already normalized, normalize high resolution data
    if norm and (np.min(HR_data) != -1 or np.max(HR_data) != 1):
        HR_data = normalize(HR_data)

    target_nii = nib.load(target_file)
    target_data = target_nii.get_fdata()

    patch_indices, ILR_patches = random_patches_3D_with_indices(ILR_data, patch_size, n_patch)

    HR_patches = []
    ground_truth_patches = []
    for index in patch_indices:
        patch = HR_data[index[0]:index[0]+patch_size[0], index[1]:index[1]+patch_size[1], index[2]:index[2]+patch_size[2], ...]
        HR_patches.append(patch)

        patch = target_data[index[0]:index[0]+patch_size[0], index[1]:index[1]+patch_size[1], index[2]:index[2]+patch_size[2], ...]
        ground_truth_patches.append(patch)

    # Convert the patches to NumPy array
    ground_truth_patches = np.array(ground_truth_patches)
    HR_patches = np.array(HR_patches)

    return patch_indices, ILR_patches, HR_patches, ground_truth_patches

def sliding_window_3d_with_indices(arr, window_shape, stride = None):
    """
    Extract patches from a whole 3D array using sliding window with specified window shape and stride.

    Args:
        arr (ndarray): 3D array.
        window_shape (tuple): Size of the sliding window.
        stride (tuple): Stride for the sliding window. If not specified, half of the window size is used.

    Returns:
        tuple: A tuple containing the patch indices and the extracted patches.
    """
    arr_shape = arr.shape
    if stride is None : stride = tuple(np.floor(np.array(window_shape)/2).astype(np.int))
    window_shape = tuple(window_shape)

    indices = []
    patches = []

    for i in range(0, arr_shape[0] - window_shape[0] + 1, stride[0]):
        for j in range(0, arr_shape[1] - window_shape[1] + 1, stride[1]):
            for k in range(0, arr_shape[2] - window_shape[2] + 1, stride[2]):
                window_indices = (i, j, k)
                window = arr[i:i+window_shape[0], j:j+window_shape[1], k:k+window_shape[2], ...]
                indices.append(window_indices)
                patches.append(window)

    return np.array(indices), np.array(patches)

def generate_3D_sliding_window_patches(patch_size, ILR_file, HR_file, target_file = None, stride = None):
    """
    Generate patches from interpolated low-resolution (ILR) and high-resolution (HR) images using sliding window, along with their ground truth patches.

    Args:
        patch_size (tuple): Size of the patches to be generated.
        ILR_file (str): File path of the low-resolution image.
        HR_file (str): File path of the high-resolution image.
        target_file (str): File path of the target image (ground truth).
        stride (tuple): Stride for the sliding window. If not specified, half of the patch size is used.

    Returns:
        tuple: A tuple containing the patch indices, ILR patches, HR patches, and ground truth patches.
    """

    # New implementation is untested
    ILR_nii = nib.load(ILR_file)
    ILR_data = ILR_nii.get_fdata()

    HR_nii = nib.load(HR_file)
    HR_data = HR_nii.get_fdata()
    if len(np.shape(HR_data))> 3 : 
        HR_data = HR_data[:,:,:,0]

    if target_file is not None: 
        target_nii = nib.load(target_file)
        target_data = target_nii.get_fdata()

    patch_indices, ILR_patches = sliding_window_3d_with_indices(ILR_data, patch_size, stride = stride)

    HR_patches = []
    ground_truth_patches = []
    for index in patch_indices:
        patch = HR_data[index[0]:index[0]+patch_size[0], index[1]:index[1]+patch_size[1], index[2]:index[2]+patch_size[2], ...]
        HR_patches.append(patch)

        if target_file is not None : 
            patch = target_data[index[0]:index[0]+patch_size[0], index[1]:index[1]+patch_size[1], index[2]:index[2]+patch_size[2], ...]
            ground_truth_patches.append(patch)

    # Convert the patches to NumPy array
    ground_truth_patches = np.array(ground_truth_patches) # Returns empty array when target file not specified
    HR_patches = np.array(HR_patches)

    return patch_indices, ILR_patches, HR_patches, ground_truth_patches


def reconstruct_image_with_overlap(patches, image_shape, patch_size, stride=None):
    """
    Reconstructs an image from overlapping patches.

    Args:
        patches (ndarray): Array of patches.
        image_shape (tuple): Shape of the output image.
        patch_size (tuple): Size of the patches.
        stride (tuple, optional): Stride for the overlapping patches. If not specified, half of the patch size is used. Defaults to None.

    Returns:
        ndarray: Reconstructed image.
    """
    if stride is None:
        stride = tuple(np.floor(np.array(patch_size) / 2).astype(np.int))

    num_patches_z = (image_shape[0] - patch_size[0]) // stride[0] + 1
    num_patches_y = (image_shape[1] - patch_size[1]) // stride[1] + 1
    num_patches_x = (image_shape[2] - patch_size[2]) // stride[2] + 1

    patches = patches.reshape((num_patches_z, num_patches_y, num_patches_x, patch_size[0], patch_size[1], patch_size[2]))

    reconstructed_image = np.zeros(image_shape, dtype=patches.dtype)
    overlap_counts = np.zeros(image_shape, dtype=np.int)

    for i in range(num_patches_z):
        for j in range(num_patches_y):
            for k in range(num_patches_x):
                z_start = i * stride[0]
                z_end = z_start + patch_size[0]
                y_start = j * stride[1]
                y_end = y_start + patch_size[1]
                x_start = k * stride[2]
                x_end = x_start + patch_size[2]

                patch = patches[i, j, k]
                reconstructed_image[z_start:z_end, y_start:y_end, x_start:x_end] += patch
                overlap_counts[z_start:z_end, y_start:y_end, x_start:x_end] += 1

    reconstructed_image /= overlap_counts

    return reconstructed_image


if __name__ == "__main__":
    # Example usage
    ILR_file = '/home/magic-chusj-2/Documents/E2022/CNN-Data/sub-010320/ses-01/anat/sub-010320_ses-01_acq-localfield_ILR.nii.gz'
    HR_file = '/home/magic-chusj-2/Documents/E2022/CNN-Data/sub-010320/ses-01/anat/sub-010320_ses-01_acq-mag_GRE_ss.nii.gz'
    target_file = '/home/magic-chusj-2/Documents/E2022/CNN-Data/sub-010320/ses-01/anat/sub-010320_ses-01_acq-localfield.nii.gz'

    # Generate random patches with corresponding indices for input and output data
    patch_size = [25, 25, 25]
    n = 3200
    patch_indices, ILR_patches, HR_patches, ground_truth_patches = generate_3D_sliding_window_patches(ILR_file, HR_file, target_file, patch_size, stride = None)