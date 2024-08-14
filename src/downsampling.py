"""
File: downsampling.py
Description: This script contains functions for image downsampling and resizing using the NIfTI file format,
             including space-invariant blurring, linear downsampling, bicubic interpolation,
             and additional utility functions for working with NIfTI images.
Dependencies: nibabel, numpy, scipy, scikit-image, os, tqdm
"""
import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.transform import resize
from nibabel.processing import fwhm2sigma, smooth_image, resample_to_output
import os
from tqdm import tqdm

def apply_space_invariant_blurring(data, fwhm):
    """
    Applies space-invariant blurring to the input 3D image data.

    Parameters:
    - data (ndarray): Input 3D image data.
    - fwhm (float): Full-width-at-half-maximum (FWHM) for Gaussian blurring.

    Returns:
    - ndarray: Blurred image data.
    """
    # Create a Gaussian filter with the specified FWHM
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    blurred_data = ndimage.gaussian_filter(data, sigma=sigma)

    return blurred_data

def linear_downsampling(data, scale_factor):
    """
    Performs linear downsampling on the input 3D image data.

    Parameters:
    - data (ndarray): Input 3D image data.
    - scale_factor (int): Downsampling scale factor.

    Returns:
    - ndarray: Downsampled image data.
    """
    # Perform linear downsampling
    downsampled_data = data[::scale_factor, ::scale_factor, ::scale_factor]

    return downsampled_data

def bicubic_interpolation_3D(image, factor):
    """
    Performs bicubic interpolation on the input 3D image.

    Parameters:
    - image (ndarray): Input 3D image data.
    - factor (float): Interpolation factor.

    Returns:
    - ndarray: Interpolated image data.
    """
    # Calculate the new shape
    new_shape = np.array(image.shape) * factor

    # Perform bicubic interpolation
    interpolated_image =  interpolated_image = resize(image, new_shape, mode='reflect', order=3, anti_aliasing=True, preserve_range=True)
    return interpolated_image

def downsample_interpolate_image(input_path, fwhm, scale_factor, save_ILR = True, output_path_ILR = None, save_LR = False, output_path_LR = None):
    """
    Loads a NIfTI image, applies space-invariant blurring, performs linear downsampling,
    and optionally saves both low-resolution (LR) and interpolated low-resolution (ILR) images.

    Parameters:
    - input_path (str): Path to the input NIfTI image.
    - fwhm (float): Full-width-at-half-maximum (FWHM) for blurring.
    - scale_factor (int): Downsampling scale factor.
    - save_ILR (bool): Flag to save the interpolated low-resolution image (default=True).
    - output_path_ILR (str): Output path for the interpolated low-resolution image.
    - save_LR (bool): Flag to save the low-resolution image (default=False).
    - output_path_LR (str): Output path for the low-resolution image.

    Returns:
    - str: Output path of the interpolated low-resolution image.
    """
    # Load the NIfTI image
    img = nib.load(input_path)
    data = img.get_fdata()

    # Apply space-invariant blurring
    blurred_data = apply_space_invariant_blurring(data, fwhm)

    # Perform linear downsampling
    downsampled_data = linear_downsampling(blurred_data, scale_factor)

    # Perform bicubi interpolation 
    upsampled_data = bicubic_interpolation_3D(downsampled_data, scale_factor)

    if save_LR : 
        if output_path_LR is None:
            output_path_LR = input_path.replace('.nii', '_LR.nii')

        # Create a new affine matrix with the same world coordinates
        new_affine = np.copy(img.affine)
        new_affine[:3, :3] /= scale_factor  # Scale the affine matrix by the downsampling factor

        # Create a new NIfTI image using the downsampled image array and updated affine matrix
        LR_img = nib.Nifti1Image(downsampled_data, new_affine, header=img.header)

        LR_img.header['pixdim'][1:4] = img.header['pixdim'][1:4] * 2  # Adjust the voxel dimensions

        # Save the downsampled image to a new file
        nib.save(LR_img, output_path_LR)
    
    if save_ILR : 
        if output_path_ILR is None:
            output_path_ILR = input_path.replace('.nii', '_ILR.nii')

        # Create a new NIfTI image with interpolated downsampled data
        ILR_img = nib.Nifti1Image(upsampled_data, img.affine, img.header) #ILR : Interpolated Low Resolution

        # Save the downsampled image to a new file
        nib.save(ILR_img, output_path_ILR)

    return output_path_ILR

def downsample_interpolate_image_nibabel(input_path, sigma = 2, lowres_vsz = [2,2,2] , save_ILR = True, output_path_ILR = None, save_LR = False, output_path_LR = None):
    """
    Loads a NIfTI image, applies blurring, performs downsampling, and optionally saves both low-resolution (LR) and interpolated low-resolution (ILR) images.

    Parameters:
    - input_path (str): Path to the input NIfTI image.
    - sigma (float): Standard deviation for blurring (default=2).
    - lowres_vsz (list): Target voxel size for downsampling (default=[2, 2, 2]).
    - save_ILR (bool): Flag to save the interpolated low-resolution image (default=True).
    - output_path_ILR (str): Output path for the interpolated low-resolution image.
    - save_LR (bool): Flag to save the low-resolution image (default=False).
    - output_path_LR (str): Output path for the low-resolution image.

    Returns:
    - str: Output path of the interpolated low-resolution image.
    """
    # Load the NIfTI image
    img = nib.load(input_path)
    initial_vsz = img.header.get_zooms()

    # Apply blurring
    if sigma!=0:
        fwhm = fwhm2sigma(sigma)
        img = smooth_image(img, fwhm)

    # Perform downsampling
    downsampled_img = resample_to_output(img, lowres_vsz)

    # Perform bicubi interpolation 
    upsampled_img = resample_to_output(downsampled_img, initial_vsz)

    if save_LR : 
        if output_path_LR is None:
            output_path_LR = input_path.replace('.nii', '_LR.nii')

        # Save the downsampled image to a new file
        nib.save(downsampled_img, output_path_LR)
    
    if save_ILR : 
        if output_path_ILR is None:
            output_path_ILR = input_path.replace('.nii', '_ILR.nii')

        # Save the downsampled image to a new file
        nib.save(upsampled_img, output_path_ILR)

    return output_path_ILR

def interpolate_image_to_target(low_res_img_file, target_voxel_sz, output_file):
    """
    Interpolates a low-resolution NIfTI image to match a target voxel size.

    Parameters:
    - low_res_img_file (str): The file path to the low-resolution NIfTI (.nii) image.
    - target_voxel_sz (list): The target voxel size for interpolation.
    - output_file (str): The file path to save the interpolated image.

    Notes:
    - The function loads the low-resolution NIfTI image from the specified file path using nibabel.
    - It performs interpolation using the target voxel size with the resample_to_output function.
    - The interpolated image is saved to the specified output file.
    """
    # Load the low-resolution NIfTI image from the specified file path
    img = nib.load(low_res_img_file)

    # Perform interpolation to match the target voxel size
    interpolated_image = nib.processing.resample_to_output(img, voxel_sizes=target_voxel_sz)

    # Save the interpolated image to the specified output file
    nib.save(interpolated_image, output_file)
    

def perform_downsampling_nibabel(input_path, sigma = 2, lowres_vsz = [2,2,2] , save_ILR = True, output_path_ILR = None, save_LR = False, output_path_LR = None):
    """
    Loads a NIfTI image, applies blurring, performs downsampling, and optionally saves both low-resolution (LR) and interpolated low-resolution (ILR) images.

    Parameters:
    - input_path (str): Path to the input NIfTI image.
    - sigma (float): Standard deviation for blurring (default=2).
    - lowres_vsz (list): Target voxel size for downsampling (default=[2, 2, 2]).
    - save_ILR (bool): Flag to save the interpolated low-resolution image (default=True).
    - output_path_ILR (str): Output path for the interpolated low-resolution image.
    - save_LR (bool): Flag to save the low-resolution image (default=False).
    - output_path_LR (str): Output path for the low-resolution image.

    Returns:
    - str: Output path of the interpolated low-resolution image.
    """
    # Load the NIfTI image
    img = nib.load(input_path)
    initial_vsz = img.header.get_zooms()

    # Apply blurring
    if sigma!=0:
        fwhm = fwhm2sigma(sigma)
        img = smooth_image(img, fwhm)

    # Perform downsampling
    downsampled_img = resample_to_output(img, lowres_vsz)

    # Perform bicubi interpolation 
    upsampled_img = resample_to_output(downsampled_img, initial_vsz)

    if save_LR : 
        if output_path_LR is None:
            output_path_LR = input_path.replace('.nii', '_LR.nii')

        # Save the downsampled image to a new file
        nib.save(downsampled_img, output_path_LR)
    
    if save_ILR : 
        if output_path_ILR is None:
            output_path_ILR = input_path.replace('.nii', '_ILR.nii')

        # Save the downsampled image to a new file
        nib.save(upsampled_img, output_path_ILR)

    return output_path_ILR

def resize_files_to_target(img_files, target_vsz, ofiles, order =3, rescale_intensity=False):
    """
    Resizes a list of NIfTI image files to a target voxel size and optionally rescales intensity.

    Parameters:
    - img_files (list): List of input NIfTI image files.
    - target_vsz (list): Target voxel size for resizing.
    - ofiles (list): Output paths for the resized images.
    - order (int): Interpolation order for resizing (default=3).
    - rescale_intensity (bool): Flag to rescale intensity (default=False).
    """
    progress_bar = tqdm(img_files, desc=f"Resizing data to {target_vsz} mm", unit="file(s)", ncols=120)
    for i, img_file in enumerate(img_files):
        progress_bar.set_postfix({"Current File": os.path.basename(img_file)})
        img = nib.load(img_file)
        downsampled_img = resample_to_output(img, target_vsz, order = order)

        # Optionally rescale intensity to match input range
        if rescale_intensity:
            # Get the intensity range of the input image
            input_min = np.min(img.get_fdata())
            input_max = np.max(img.get_fdata())
            
            # Get the intensity range of the downsampled image
            output_min = np.min(downsampled_img.get_fdata())
            output_max = np.max(downsampled_img.get_fdata())
            
            # Rescale the intensities of the downsampled image to match the input range
            downsampled_img_data = downsampled_img.get_fdata()
            downsampled_img_data = (downsampled_img_data - output_min) / (output_max - output_min)
            downsampled_img_data = downsampled_img_data * (input_max - input_min) + input_min
            
            # Create a new NIfTI image with the rescaled data
            downsampled_img = nib.Nifti1Image(downsampled_img_data, downsampled_img.affine)

        nib.save(downsampled_img, ofiles[i])
        progress_bar.update(1)
    progress_bar.close()

if __name__ == '__main__': 
    # Example usage 1 
    # input_path = '/home/magic-chusj-2/Documents/E2022/CNN-Data/sub-010036/ses-01/anat/sub-010036_ses-01_acq-localfield.nii.gz'
    # # output_path = input_path.replace('localfield', 'localfield_ILR')
    # fwhm = 0.8  # Full-width-at-half-maximum (FWHM) for blurring, equal to the slice thickness
    # scale_factor = 2  # Downsampling scale factor (e.g., 2 for halving the resolution)

    # downsample_interpolate_image(input_path, fwhm, scale_factor, save_LR = True)
    # basepath = '/home/magic-chusj-2/Documents/E2022/downsampled_dataset'
    # LR_path = '1.9mm'
    # HR_path = '1.05mm'
    # HR_msk_files = sorted(glob.glob(os.path.join(basepath, HR_path, '*/*mask*.nii*')))
    # LR_mag_files = sorted(glob.glob(os.path.join(basepath, LR_path, '*/*e1.nii')))
    # for HR_msk, LR_mag in zip(HR_msk_files, LR_mag_files):
    #     ofile = LR_mag.replace('e1.nii', 'mask.nii')
    #     downsample_to_target_size(HR_msk, LR_mag, ofile)

    # Example usage 2
    # Bluring + downsampling + upsampling training data test
    input_path = '/home/magic-chusj-2/Documents/E2022/CNN-Data/sub-010036/ses-01/anat/sub-010036_ses-01_acq-localfield.nii.gz'
    sigma = 2
    downsample_interpolate_image_nibabel(input_path, save_ILR = True,save_LR = True)