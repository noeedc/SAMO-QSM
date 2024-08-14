"""
QSM Preprocessing Pipeline

This script performs various preprocessing steps on MRI data, including bias field correction,
skull stripping, phase unwrapping, background field removal, downsampling, and interpolation.
It also includes functions for creating masks, performing registration, and generating a MATLAB
workspace file for QSM reconstruction.

Requirements:
- N4BiasFieldCorrection (from ANTs software)
- FSL (FMRIB Software Library)
- MATLAB with STI Suite v3.0


Author: Noée Ducros-Chabot
Date: 1st June 2023
Last Modified : 22th November 2023
"""
import os 
import glob
import numpy as np 
import matlab.engine
import nibabel as nib
from tqdm import tqdm
import warnings
from scipy.io import savemat
from scipy.spatial.transform import Rotation
import ants

from src.normalize import normalize_pi, normalize, apply_norm_pi
from src.downsampling import downsample_interpolate_image_nibabel, resize_files_to_target, interpolate_image_to_target
from src.crop_foreground import crop_nifti_with_nilearn
from src.padding import pad_to_even_slices_3d, remove_padding_3d
from src.register import ants_rigid_registration, ants_apply_single_transform
from src.nifti_file_tools import get_voxel_size
from config.paths import MEDI_PATH, STISuite_path


def apply_mask_and_save(mask_file, magnitude_file, output_name):
    # TO DO : output_file as optional argument
    """
    Applies a binary mask to a magnitude NIfTI image and saves the result.

    Parameters:
    - mask_file (str): The file path to the binary mask NIfTI (.nii) file.
    - magnitude_file (str): The file path to the magnitude NIfTI (.nii) file.
    - output_name (str): The desired name for the output masked magnitude NIfTI image.

    Notes:
    - The function loads the binary mask and magnitude NIfTI images from the specified file paths using nibabel.
    - It extracts the voxel data as NumPy arrays and applies the binary mask to the magnitude data.
    - A new NIfTI image is created with the masked magnitude data, using the affine transformation
      and header information from the original magnitude image.
    - The resulting masked magnitude image is saved to the specified output file with the given name.
    """
    # Load the binary mask NIfTI image from the specified file path
    mask_img = nib.load(mask_file)

    # Load the magnitude NIfTI image from the specified file path
    magnitude_img = nib.load(magnitude_file)

    # Get the voxel data arrays from the mask and magnitude images
    mask_data = mask_img.get_fdata().astype(np.bool)
    magnitude_data = magnitude_img.get_fdata()

    # Apply the binary mask to the magnitude data
    masked_magnitude_data = np.where(mask_data, magnitude_data, 0)

    # Create a new NIfTI image with the masked magnitude data, using the original affine transformation and header
    masked_magnitude_img = nib.Nifti1Image(masked_magnitude_data, affine=magnitude_img.affine, header=magnitude_img.header)

    # Save the resulting masked magnitude image to the specified output file with the given name
    nib.save(masked_magnitude_img, output_name)

def fsl_mask_creation(input_file, output_file = None, f = 0.5, g = 0 ):
    if output_file == None :
        output_file = input_file.replace('.nii', '')
    os.system(f'bet2 "{input_file}" "{output_file}" -m -f {f} -g {g}')
    o_msk = input_file.replace(".nii", "_mask.nii.gz")
    return o_msk

def perform_bias_field_correction(mag_files, mag_suffix='mag_GRE', folder_name_idx=-2, msk_creation=True, msk_files=[]):
    """
    Performs bias field correction on a set of magnitude NIfTI images and optionally creates skull-stripped images.

    Parameters:
    - mag_files (list): A list of file paths to magnitude NIfTI (.nii) files.
    - mag_suffix (str): The suffix in the file names to identify the magnitude images (default: 'mag_GRE').
    - folder_name_idx (int): The index of the folder name in the file path (default: -2).
    - msk_creation (bool): Flag indicating whether to create skull-stripped images (default: True).
    - msk_files (list): A list of file paths to mask NIfTI (.nii) files used for skull stripping (default: []).

    Returns:
    - ssMag_files (list): A list of file paths to the skull-stripped magnitude images.
    - ouput_msk_files (list): A list of file paths to the created skull-stripped mask images.

    Notes:
    - The function performs bias field correction using the N4BiasFieldCorrection algorithm from ANTs.
    - Optionally, it creates skull-stripped images using the Brain Extraction Tool (BET) from FSL.
    - The input and output file paths are derived based on the specified suffix and folder index.
    - If mask files are provided, they are used for skull stripping; otherwise, skull stripping is skipped.
    """
    ouput_msk_files = []  # Masks will be created if not passed as input
    ssMag_files = []  # Skull-stripped magnitude files

    if len(msk_files) != 0:
        msk_files = sorted(msk_files)  # Ensure files are sorted so the index corresponds to the right mask

    # Create a tqdm progress bar
    progress_bar = tqdm(mag_files, desc="Bias Field Correction and Mask Creation", unit="file(s)", ncols=150)

    for i, f in enumerate(mag_files):
        folder_name = f.split('/')[folder_name_idx]
        progress_bar.set_postfix({"Current Folder": folder_name})

        o_corrected = f.replace(mag_suffix, 'mag_corrected')  # Magnitude-corrected image (with skull)
        o_msk = f.replace(mag_suffix, 'mag_ss')  # 'ss' for skull-stripped
        if ".gz" not in o_msk:
            o_msk = o_msk.replace(".nii", ".nii.gz")
        ssMag_files.append(o_msk)

        d = len(nib.load(f).shape)  # TO DO: Test with 3D image

        os.system(f'N4BiasFieldCorrection -d {d} -i "{f}" -o "{o_corrected}"')

        if msk_creation:
            # TO DO : Remove this option
            os.system(f'bet "{o_corrected}" "{o_msk}" -m')
            o_msk = o_msk.replace(".nii", "_mask.nii")
            ouput_msk_files.append(o_msk)
        else:
            # Create skull-stripped image
            if len(msk_files) == 0:
                warnings.warn('Mask files not passed as input. Unable to create skull-stripped image with '
                              'bias field-corrected magnitude.')
            else:
                apply_mask_and_save(msk_files[i], o_corrected, o_msk)

        # Update the progress bar
        progress_bar.update(1)

    progress_bar.close()
    return ssMag_files, ouput_msk_files


def perform_phase_unwrapping_and_background_removal(phs_files, msk_files, voxelsizes, TEs,
                                                    STISuite_path= STISuite_path,
                                                    padsize=[12, 12, 12], smvsize=12, phs_suffix='phase_GRE'):
    """
    Performs phase unwrapping and background removal on a set of phase NIfTI images.

    Parameters:
    - phs_files (list): A list of file paths to phase NIfTI (.nii) files.
    - msk_files (list): A list of file paths to mask NIfTI (.nii) files corresponding to the phase images.
    - voxelsizes (list or float): List of voxel sizes or a single voxel size value for all orientations.
    - TEs (list): List of echo times corresponding to the phase images.
    - STISuite_path (str): The path to the STI Suite v3.0 toolbox (default: from config file).
    - padsize (list): The padding size for Laplacian phase unwrapping (default: [12, 12, 12]).
    - smvsize (int): The smvsize parameter for V-SHARP background removal (default: 12).
    - phs_suffix (str): The suffix in the file names to identify the phase images (default: 'phase_GRE').

    Returns:
    - localfield_files (list): A list of file paths to the resulting local field images.

    Notes:
    - The function performs phase unwrapping using the MRPhaseUnwrap algorithm and background removal using V-SHARP
      from the STI Suite v3.0 toolbox.
    - Input and output file paths are derived based on the specified suffix.
    - Voxelsizes can be either a list of voxel sizes or a single voxel size value for all orientations.
    """
    tolerance = 1e-6
    # Prep matlab engine
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(STISuite_path))

    # If voxel size contains values instead of a list, it means voxel sizes are the same for all orientations
    if type(voxelsizes[0]) != list:
        voxelsize = voxelsizes * len(phs_files)

    if len(voxelsizes) != len(phs_files):
        last_voxel_size = voxelsizes[-1]
        voxelsizes.extend([last_voxel_size] * (len(phs_files) - len(voxelsizes)))
        print("Warning: Adjusted voxel_sizes to match the number of phs_files.")

    localfield_files = []

    # Make sure there is a correspondence between mask and phase data
    for phs_f, msk_f, voxelsize in zip(phs_files, msk_files, voxelsizes):
        print(f'\nProcessing folder: {os.path.basename(os.path.dirname(phs_f))}')

        phs_nii = nib.load(phs_f)
        phs_data = phs_nii.dataobj

        # If the range of the image is not already [-π, π)
        if not (-np.pi - tolerance <= np.min(phs_data) and np.max(phs_data) <= np.pi + tolerance):
            # Normalize wrapped phase intensities between [-π, π)
            norm_img = apply_norm_pi(phs_f, output_suffix='_norm')
        else:
            norm_img = phs_nii.get_fdata()

        # Mask phase image
        msk = nib.load(msk_f).get_fdata()
        msk_dtype = msk.dtype
        msk = msk.astype(bool)

        voxelsize = np.round(voxelsize, 3)

        # Check if image shape is even or if padding is necessary
        padding_necessary = not (np.all(np.array(norm_img.shape) % 2 == 0))
        if padding_necessary:
            norm_img, padding = pad_to_even_slices_3d(norm_img)

        # Laplacian unwrapping code from STI Suite v3.0
        try:
            fieldmap = np.asarray(
                eng.MRPhaseUnwrap(matlab.double(norm_img), 'voxelsize', matlab.single(voxelsize), 'padsize',
                                  matlab.double(padsize), nargout=1))
        except:
            fieldmap = np.asarray(
                eng.MRPhaseUnwrap(matlab.double(norm_img), 'voxelsize', matlab.double(voxelsize), 'padsize',
                                  matlab.double(padsize), nargout=1))

        if padding_necessary:
            fieldmap = remove_padding_3d(fieldmap, padding)

        # Divide by delta TE and by TE if delta TE does not exist
        fieldmap = fieldmap / TEs[0] / (2 * np.pi)  # conversion in Hz
        fieldmap = fieldmap * msk

        # Save unwrapped image
        o_fmap = phs_f.replace(phs_suffix, 'fieldmap')
        fieldmap_nii = nib.Nifti1Image(fieldmap, phs_nii.affine, phs_nii.header)
        nib.save(fieldmap_nii, o_fmap)

        # VSharp Background field removal from STI-Suite v3.0
        try:
            tissuePhase, nMsk = eng.V_SHARP(fieldmap, msk, 'voxelsize', matlab.single(voxelsize), 'smvsize',
                                            matlab.double(smvsize), nargout=2)
        except:
            tissuePhase, nMsk = eng.V_SHARP(fieldmap, msk, 'voxelsize', matlab.double(voxelsize), 'smvsize',
                                            matlab.double(smvsize), nargout=2)
        tissuePhase, nMsk = np.asarray(tissuePhase), np.asarray(nMsk)

        # Save local field and new mask
        if phs_suffix in phs_f:
            o_localfield = phs_f.replace(phs_suffix, 'localfield')
            o_nmsk = phs_f.replace(phs_suffix, 'bckremoval_msk')
        else:
            warnings.warn(
                f"Phase suffix: {phs_suffix} does not appear to be present in the phase file. Saving name directly.")
            folder_path = os.path.dirname(phs_suffix)
            o_localfield = os.path.join(folder_path, 'localfield.nii.gz')
            o_nmsk = os.path.join(folder_path, 'bckremoval_msk.nii.gz')
        localfield_files.append(o_localfield)
        localfield_nii = nib.Nifti1Image(tissuePhase, phs_nii.affine, phs_nii.header)
        nib.save(localfield_nii, o_localfield)
        nMsk_nii = nib.Nifti1Image(nMsk.astype(msk_dtype), phs_nii.affine, phs_nii.header)
        nib.save(nMsk_nii, o_nmsk)

    return localfield_files


def perform_downsampling_and_interpolation(localfield_files, lowres_voxelsize=[2, 2, 2], sigma=2):
    """
    Performs downsampling and interpolation on a set of local field NIfTI images.

    Parameters:
    - localfield_files (list): A list of file paths to local field NIfTI (.nii) files.
    - lowres_voxelsize (list): The target voxel size for downsampling (default: [2, 2, 2]).
    - sigma (float): The standard deviation of the Gaussian kernel for interpolation (default: 2).

    Returns:
    - ILR_files (list): A list of file paths to the resulting low-resolution (ILR) images.

    Notes:
    - The function utilizes the downsample_interpolate_image_nibabel function to perform downsampling and interpolation.
    - Input and output file paths are derived based on the specified parameters.
    """
    # Create a tqdm progress bar
    progress_bar = tqdm(localfield_files, desc="Preparing Low-Resolution Data", unit="file(s)", ncols=120)

    ILR_files = []

    for localfield_f in localfield_files:
        # Update the progress bar
        folder_name = localfield_f.split('/')[6]
        progress_bar.set_postfix({"Current Folder": folder_name})

        # Perform downsampling and interpolation using the specified parameters
        ILR_file = downsample_interpolate_image_nibabel(localfield_f, sigma=sigma, lowres_vsz=lowres_voxelsize,
                                                         save_LR=True, save_ILR=True)
        ILR_files.append(ILR_file)

        # Update the progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    return ILR_files


def perform_resizing(localfield_files, target_vsz, output_files=None):
    """
    Performs interpolation on a set of local field NIfTI images to match a target voxel size.

    Parameters:
    - localfield_files (list): A list of file paths to local field NIfTI (.nii) images.
    - target_vsz (list): The target voxel size for interpolation.
    - output_files (list or None): Optional list of file paths to save the interpolated images.
                                   If None, new file names will be generated. (default: None)

    Returns:
    - output_files (list): A list of file paths to the resulting interpolated images.

    Notes:
    - The function uses the interpolate_image_to_target function to perform interpolation.
    - If output_files is provided, it will be used for saving the interpolated images;
      otherwise, new file names will be generated.
    """
    # Create a tqdm progress bar
    progress_bar = tqdm(localfield_files, desc="Interpolating Low-Resolution Data", unit="file(s)", ncols=120)

    # Check if output_files is provided; if not, create new file names
    create_ofiles = output_files is None
    if create_ofiles:
        output_files = []

    for i, localfield_f in enumerate(localfield_files):
        # Update the progress bar
        folder_name = os.path.basename(os.path.dirname(localfield_f))
        progress_bar.set_postfix({"Current Folder": folder_name})

        # Generate output file path
        if create_ofiles:
            ofile = localfield_f.replace('.nii', '_ILR.nii')
        else:
            ofile = output_files[i]

        # Perform interpolation using the target voxel size
        interpolate_image_to_target(localfield_f, target_vsz, ofile)

        # Append output file path to the list
        if create_ofiles:
            output_files.append(ofile)

        # Update the progress bar
        progress_bar.update(1)

    # Close the progress bar
    progress_bar.close()

    return output_files


def perform_echo_combination(localfield_files, magn_files, MEDI_path = MEDI_PATH, folder_name_idx = 6):
    """
    Combines multiple temporal echos using MEDI non-linear fit and saves the results in NIfTI format.

    Parameters:
        localfield_files (list of str): List of file paths containing the local field maps.
        magn_files (list of str): List of file paths containing the magnitude images.
        MEDI_path (str): Path to the MEDI toolbox. Default is '/home/magic-chusj-2/Documents/E2022/QSM/QSM-MEDI_toolbox'.
        folder_name_idx (int): Index of the folder name containing the subject ID in the file path. Default is 6.

    Returns:
        list of str: List of file paths containing the echo-combined local field maps.
    """
    # Prep matlab engine
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath(MEDI_path))

    progress_bar = tqdm(localfield_files, desc="Combining temporal echos", unit="file(s)", ncols=120)
    echoCombined_phs_files = []
    for localfile, magn_file in zip(localfield_files, magn_files):
        folder_name = localfile.split('/')[folder_name_idx]
        progress_bar.set_postfix({"Current Folder": folder_name})

        fieldMap_nii = nib.load(localfile)
        fieldMap = fieldMap_nii.get_fdata()
        if fieldMap.max()> np.pi or fieldMap.min() < -np.pi: fieldMap = normalize_pi(fieldMap)

        # Echo phase combination : MEDI non linear fit       
        magn = nib.load(magn_file).get_fdata()
        iFreq_raw, _ = eng.Fit_ppm_complex(magn * np.exp(-1j * fieldMap),  nargout=2)
        iFreq_raw = np.asarray(iFreq_raw)

        # Save echo combined img
        o_file = localfile.replace('.nii', '_echoCombined.nii')
        echoCombined_phs_files.append(o_file)

        echoCombined_nii = nib.Nifti1Image(iFreq_raw, fieldMap_nii.affine, fieldMap_nii.header)
        nib.save(echoCombined_nii, o_file)

        progress_bar.update(1)
    progress_bar.close()

    return echoCombined_phs_files


def normalize_mag_files(files, min = 0, max = 0.15, suffix = '_norm'):
    """
    Normalize magnitude images and save the results in NIfTI format.

    Parameters:
        files (list of str): List of file paths containing the magnitude images.
        min (float): Minimum value for normalization. Default is 0.
        max (float): Maximum value for normalization. Default is 0.15.
        suffix (str): Suffix to be added to the output file names. Default is 'norm'.

    Returns:
        list of str: List of file paths containing the normalized magnitude images.
    """
    norm_mag = []
    for file in files :
        mag_nii = nib.load(file)
        mag = mag_nii.get_fdata()
        mag = normalize(mag, new_min = min, new_max = max)

        new_file = file.replace('.nii', f'{suffix}.nii')
        norm_mag_nii = nib.Nifti1Image(mag, mag_nii.affine, mag_nii.header)
        nib.save(norm_mag_nii, new_file)
        norm_mag.append(new_file)
    return norm_mag


def create_matlab_workspace(data_folder, agg_mag, msk, agg_localfield, rot_img, msk_file, ofile_name, TEs, voxelsize, B0 = 3.0, CF = 123.26, gyro = 42.58):
    """
    Creates a MATLAB workspace file (.mat) containing parameters and data for QSM reconstruction.

    Parameters:
    - data_folder (str): The path to the data folder.
    - agg_mag (numpy.ndarray): The aggregated magnitude image data.
    - msk (numpy.ndarray): The mask image data.
    - agg_localfield (numpy.ndarray): The aggregated local field image data.
    - rot_img (numpy.ndarray): The rotated image data.
    - msk_file (str): The file path to the mask NIfTI (.nii) file.
    - ofile_name (str): The desired name for the MATLAB workspace file.
    - TEs (list): List of echo times corresponding to the magnitude images.
    - voxelsize (list): The voxel size of the images.

    Returns:
    - workspace_ofile (str): The file path to the created MATLAB workspace file.

    Notes:
    - The function converts the local field values to parts per million (ppm) if necessary.
    - The MATLAB workspace file includes parameters such as B0, CF, gyro, phs_scale, dim, N, TEs, vsz, C, FL_all, MAG_all,
      R_TOT, msk, directory, and nifti_file.
    """
    phs_scale = B0 * gyro

    N = np.array(msk.shape).astype(np.float64)
    dim = np.array(agg_mag.shape).astype(np.float64)
    order = "CEFLR"  # order of the orientation folder 'Central, Extension, Flexion, Left, Right' (alphabetical)

    # PPM conversion
    if np.max(agg_localfield) > 10:
        agg_localfield = agg_localfield / phs_scale

    if np.max(agg_localfield) > 1:
        warnings.warn("Check the local field image. There might be an error in the range/ppm conversion", category=UserWarning)

    matlab_params = {"B0": B0, "CF": CF, "gyro": gyro, "phs_scale": B0 * gyro, "dim": dim, "N": N, "TEs": TEs,
                     "vsz": voxelsize, "C": order, "FL_all": agg_localfield, "MAG_all": agg_mag,
                     "R_TOT": rot_img, "msk": msk, "directory": data_folder, "nifti_file": msk_file}

    workspace_ofile = os.path.join(data_folder, ofile_name)
    savemat(workspace_ofile, matlab_params)
    return workspace_ofile

def perform_registration_and_mask_generation(central_mag, central_msk, e1_files, data_folder, generate_mask=True):
    """
    Perform rigid registration and mask generation for a list of magnitude files relative to a central magnitude file.

    Parameters:
    - central_mag (str): Path to the central magnitude file.
    - central_msk (str): Path to the central mask file.
    - e1_files (list): List of magnitude file paths to be registered.
    - data_folder (str): Root folder for storing registration and mask files.
    - generate_mask (bool): Whether to generate masks during the registration process.

    Returns:
    - transformation_files (list): List of transformation files generated during registration.
    - msk_files (list): List of mask files generated during registration.
    """

    transformation_files = []
    msk_files = []

    for i, mag_file in enumerate(e1_files):
        if i != 0:
            subfolder = os.path.basename(os.path.dirname(mag_file))

            outpath = os.path.join(data_folder, subfolder, 'central_to_complementary_reg/')
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            ofile = os.path.basename(central_mag).replace('e1.nii', 'e1_reg.nii.gz')

            if subfolder != 'results':
                ants_rigid_registration(outpath, ofile, mag_file, central_mag)
                transform_file = os.path.join(outpath, 'Composite.h5')
                transformation_files.append(transform_file)

                if generate_mask:
                    omsk_file = os.path.join(outpath, os.path.basename(central_mag).replace('e1.nii', 'mask.nii.gz'))
                    ants_apply_single_transform(transform_file, central_msk, mag_file, omsk_file, interpolate='NearestNeighbor')

                    msk_files.append(omsk_file)
                    print(f"Generated a mask for '{subfolder}' by registering the central image's mask.")
                    print(f"The mask has been saved as '{omsk_file}'.\n")

    return transformation_files, msk_files

def perform_localfield_registration_and_interpolation(transformation_files, localfield_files, central_mag, data_folder):
    """
    Performs registration and transformation for a list of localfield files relative to a central magnitude file.

    Parameters:
    - transformation_files (list): List of transformation files.
    - localfield_files (list): List of localfield file paths to be registered.
    - central_mag (str): Path to the central magnitude file.
    - data_folder (str): Root folder for storing registration and transformation files.

    Returns:
    - reg_ILR_locafields (list): List of registered localfield files.
    - rot_matrices (numpy.ndarray): Array of rotation matrices.
    """
    n = len(localfield_files)+1 # Number of orientations
    rot_matrices = np.zeros((3,3,n))
    rot_matrices[:,:,0] = np.identity(3)

    reg_locafields = []
    for i, trans_file, localfield in zip(range(1, n), transformation_files, localfield_files):
        subfolder = os.path.basename(os.path.dirname(localfield))

        outpath = os.path.join(data_folder, subfolder, 'complementary_to_central_reg/')
        if not os.path.exists(outpath):
            os.makedirs(outpath)

        # Save rotation matrix
        transform = ants.read_transform(trans_file)
        rot_matrix = transform.parameters.reshape((4, 3))[:3]
        r = Rotation.from_matrix(rot_matrix)
        angles = r.as_euler("xyz", degrees=True)

        rot_file = os.path.join(outpath, 'trans_matrix.txt')
        np.savetxt(rot_file, rot_matrix)
        rot_matrices[:, :, i] = rot_matrix[:3, :3]
        
        # Save euler angles
        np.savetxt(os.path.join(outpath, 'angles.txt'), angles)

        if subfolder != 'results':
            inv_trans_file = trans_file.replace('Composite', 'InverseComposite')
            name = os.path.basename(localfield).replace('.nii', '_registered.nii' )
            ofile = os.path.join(outpath, name)
            
            ants_apply_single_transform(inv_trans_file, localfield, central_mag, ofile, interpolate='BSpline') 
            # If size of target image is not the same as input image, the image is interpolated to a higher resolution.
            #ants_rigid_registration(outpath, ofile, localfield, localfield_files[0]) # Second registration is necessary 
            reg_locafields.append(ofile)

    return reg_locafields, rot_matrices

if __name__ == "__main__":
    # Example usage
    data_folder = '/home/magic-chusj-2/Documents/E2022/CNN-Data(copy)'

    mag_files = sorted(glob.glob(os.path.join(data_folder, 'sub-*/ses*/anat/*mag_GRE.nii*')))
    phs_files = sorted(glob.glob(os.path.join(data_folder, 'sub-*/ses*/anat/*phase_GRE.nii*')))

    print('\n------------- Cropping image -------------')
    progress_bar = tqdm(mag_files, desc="Cropping images", unit="file(s)", ncols=120)
    for mag_file, phs_file in zip(mag_files, phs_files):
        progress_bar.set_postfix({"Current Folder": os.path.basename(mag_file)})
        crop_nifti_with_nilearn(mag_file, [phs_file])
        progress_bar.update(1)
    progress_bar.close()

    mag_cropped_files = sorted(glob.glob(os.path.join(data_folder, 'sub-*/ses*/anat/*mag_GRE_cropped.nii*')))
    phs_cropped_files = sorted(glob.glob(os.path.join(data_folder, 'sub-*/ses*/anat/*phase_GRE_cropped.nii*')))

    print('\n------------- N4BiasFieldCorrection & fslBET -------------')
    _, msk_files = perform_bias_field_correction(mag_cropped_files)
    # msk_files = sorted(glob.glob(os.path.join(data_folder, 'sub-*/ses*/anat/*cropped_mask.nii*')))

    print('\n ---------- Phase Unwrapping & Background field removal ----------')
    vsz = get_voxel_size(mag_cropped_files[0])
    TEs = [17.3e-3]
    localfield_files = perform_phase_unwrapping_and_background_removal(phs_cropped_files, msk_files, [vsz], TEs, phs_suffix = 'phase_GRE_cropped')

    #localfield_files = sorted(glob.glob(os.path.join(data_folder, 'sub-*/ses*/anat/*localfield.nii*')))
    print('\n------------- Downsampling to 1mm isotropic -------------')
    target_vsz = [1.0,1.0,1.0]
    localfield_1mm_files = [x.replace('.nii', '_1mm.nii') for x in localfield_files]
    resize_files_to_target(localfield_files, target_vsz, localfield_1mm_files, order = 1)
    ss_mag_files = sorted(glob.glob(os.path.join(data_folder, 'sub-*/ses*/anat/*mag_ss_cropped.nii*')))
    mag_1mm_files = [x.replace('.nii', '_1mm.nii') for x in ss_mag_files]
    resize_files_to_target(ss_mag_files, target_vsz, mag_1mm_files, order = 1)
    
    eroded_msk = sorted(glob.glob(os.path.join(data_folder, 'sub-*/ses*/anat/*bckremoval_msk.nii*')))
    mask_1mm_files = [x.replace('.nii', '_1mm.nii') for x in msk_files]
    resize_files_to_target(eroded_msk, target_vsz, mask_1mm_files, order = 0)
    
    # localfield_1mm_files = sorted(glob.glob(os.path.join(data_folder, 'sub-*/ses*/anat/*localfield_1mm.nii*')))
    print('\n---------- Downsampling & Interpolation ----------')
    perform_downsampling_and_interpolation(localfield_1mm_files)

