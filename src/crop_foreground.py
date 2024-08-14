"""
File: crop_foregground.py
Author: Noée Ducros-Chabot
Description: This script provides functions for cropping NIfTI images using nibabel and Nilearn.
"""
import nibabel as nib
from nilearn import image 

def crop_nifti_with_slices(input_file_path, output_file_path, start_indices, stop_indices, target_affine = None):
    """
    Crop a NIfTI image with specified start and stop indices and save the result.

    Args:
        input_file_path (str): Path to the input NIfTI file.
        output_file_path (str): Path to save the cropped image.
        start_indices (tuple): Start indices for cropping.
        stop_indices (tuple): Stop indices for cropping.
        target_affine (ndarray, optional): Target affine matrix. Defaults to None.
    """    
    # Load the NIfTI image
    img = nib.load(input_file_path)
    img_data = img.get_fdata()

    # Crop the image using the specified start and stop indices
    cropped_img_data = img_data[start_indices[0]:stop_indices[0],
                                start_indices[1]:stop_indices[1],
                                start_indices[2]:stop_indices[2]]

    # Create a new affine matrix with centered world coordinates
    new_affine = img.affine.copy()
    new_affine[:3, 3] = img.affine[:3, 3] + start_indices @ img.affine[:3,:3] 

    # Create a new NIfTI image with the cropped data and updated affine matrix
    cropped_img = nib.Nifti1Image(cropped_img_data, new_affine)

    # Save the cropped and centered image
    nib.save(cropped_img, output_file_path)

def crop_nifti_with_nilearn(input_file_path, other_files):
    """
    Crop a NIfTI image using Nilearn and save the result. Also crop additional images with the same indices.

    Args:
        input_file_path (str): Path to the input NIfTI file.
        other_files (list): List of paths to additional NIfTI files for cropping.
    """
    img = nib.load(input_file_path)
    
    cropped_nii, offset = image.crop_img(img, rtol=1e-06, copy=True, pad=True, return_offset=True)

    start_indices = [slice_obj.start for slice_obj in offset]
    stop_indices = [slice_obj.stop for slice_obj in offset]

    # Save the cropped image
    output_file_path = input_file_path.replace('.nii', '_cropped.nii')
    nib.save(cropped_nii, output_file_path)

    for img_file in other_files:
        img = nib.load(img_file)
        output_file_path = img_file.replace('.nii', '_cropped.nii')
        crop_nifti_with_slices(img_file, output_file_path, start_indices, stop_indices)

if __name__ == '__main__':
    # Example usage
    data_path = '/home/magic-chusj-2/Documents/E2022/AMONI/HBHLNeurosphere_13_10_22_for_train'
    mag_file = '/home/magic-chusj-2/Documents/E2022/AMONI/HBHLNeurosphere_13_10_22_for_train/gre_IPAT2_pPF_sPF_3D_6echoes/nifti/gre_IPAT2_pPF_sPF_3D_6echoes_Series0004_gre_IPAT2_pPF_sPF_3D_6echoes_20221013115607_4_mag_GRE_ss_test.nii.gz'
    # all_localfields = sorted(glob.glob(os.path.join(data_path, '*/nifti/complementary_to_central_reg/*registered.nii')))
    all_localfields = ['/home/magic-chusj-2/Documents/E2022/AMONI/HBHLNeurosphere_13_10_22_for_train/gre_IPAT2_pPF_sPF_3D_6echoes/nifti/gre_IPAT2_pPF_sPF_3D_6echoes_Series0005_gre_IPAT2_pPF_sPF_3D_6echoes_20221013115607_5_localfield_ILR.nii', 
    '/home/magic-chusj-2/Documents/E2022/AMONI/HBHLNeurosphere_13_10_22_for_train/gre_IPAT2_pPF_sPF_3D_6echoes/nifti/gre_IPAT2_pPF_sPF_3D_6echoes_Series0005_gre_IPAT2_pPF_sPF_3D_6echoes_20221013115607_5_localfield.nii']
    crop_nifti_with_nilearn(mag_file, all_localfields) 
    