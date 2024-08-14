"""
Script: NIfTI Image Difference Calculation and Saving

Description:
This script computes the voxel-wise difference between two NIfTI images and saves the resulting difference image.
Optionally, it can calculate the absolute difference. The script is designed to be used with NIfTI images in the
context of medical imaging or similar domains.

Author: Noée Ducros-Chabot
Date: December 14th 2023

Dependencies:
- nibabel
- glob
- os

Usage:
1. Ensure the required library 'nibabel' is installed. You can install it using: pip install nibabel

2. Update the 'data_folder' variable with the path to the folder containing the NIfTI images for comparison.

3. Update the 'target_file' variable with the path to the reference NIfTI image.

4. Run the script to calculate the voxel-wise difference and save the resulting NIfTI images.

Note: This script assumes NIfTI images are stored in a specific structure within the 'data_folder', and the file names
end with '_CFR.nii'. The script will create difference images with filenames ending with '_diff.nii'.

"""
import nibabel as nib
import glob 
import os

def create_and_save_difference_nifti(input_nifti1, input_nifti2, output_nifti, absolute_difference=False):
    """
    Compute the difference between two NIfTI images and save the result.

    Parameters:
        input_nifti1 (str): Path to the first input NIfTI image.
        input_nifti2 (str): Path to the second input NIfTI image.
        output_nifti (str): Path to save the difference NIfTI image.
        absolute_difference (bool): If True, compute the absolute difference.

    Returns:
        None
    """

    # Load NIfTI images
    nifti1 = nib.load(input_nifti1)
    nifti2 = nib.load(input_nifti2)

    # Get image data arrays
    image1 = nifti1.get_fdata()
    image2 = nifti2.get_fdata()

    # Ensure the two input images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("Input images must have the same shape")

    # Compute the difference image
    difference_image = image1 - image2
    
    if absolute_difference:
        difference_image = abs(difference_image)

    # Create a NIfTI image object for the difference image
    difference_nifti = nib.Nifti1Image(difference_image, nifti1.affine)

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_nifti)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the difference NIfTI image
    nib.save(difference_nifti, output_nifti)

if __name__ == "__main__":
    # Example usage
    data_folder = '/home/magic-chusj-2/Documents/E2022/downsampled_dataset_nCNN/2.4mm/results'
    target_file = '/home/magic-chusj-2/Documents/E2022/downsampled_dataset/1.05mm/results/HR_localfields/w0.0_tkd0.00/sCosmos_ori5.nii'

    files = glob.glob(os.path.join(data_folder, '*/*/*_CFR.nii'))

    for file in files :
        ofile = file.replace('.nii', '_diff.nii').replace('.gz', '')
        create_and_save_difference_nifti(target_file, file, ofile, absolute_difference=False )