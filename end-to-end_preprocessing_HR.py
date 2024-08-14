#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
QSM Pipeline Script
-------------------

This script performs various pre-processing and post-processing steps for Quantitative Susceptibility Mapping (QSM)
of magnetic resonance images. It combines multiple temporal echoes using MEDI non-linear fit, performs bias field
correction, echo combination, phase unwrapping, background removal, downsampling (to simulate low resolution images), and applies a Convolutional Neural
Network (CNN) model to perform Super-Resolution on the localfield images. Finally, it aggregates the processed data and saves the results in NIfTI format and MATLAB
workspace to run COSMOS QSM reconstruction.

This script is for high resolution (HR) data. It has the option to simulate downsampled data.

Author: Noée Ducros-Chabot
Date: 2nd of August 2023

Usage: Modify the 'data_folder' variable with the path to the subject data. Make sure to have all the required
    dependencies installed and provide the correct paths to the MEDI toolbox and the CNN model.

Steps:
1. N4BiasFieldCorrection and fslBET:
   - Corrects the bias field in magnitude MRI images using N4BiasFieldCorrection.
   - Performs skull stripping on the corrected magnitude images using fslBET.

2. Phase Unwrapping and Background Field Removal:
   - Normalizes the wrapped phase intensities between [-π, π).
   - Masks the phase image using the skull-stripped mask from the previous step.
   - Performs phase unwrapping using Laplacian unwrapping algorithm from STI Suite v3.0.
   - Converts the unwrapped phase to a field map in Hz units.
   - Saves the unwrapped field map as a NIfTI file.
   - Performs VSharp background field removal using STI Suite v3.0.
   - Saves the local field (tissue phase) and the new mask after background field removal.

3. (Optional) Simulate Downsampled Data

4. Local Field Registration:
    - Applies registration of local fields from complementary images to the central image.
    - (If simluated downsampling) Bicubic interpolation is applied during the registration, augmenting the resolution of the image to the central resolution.

5. (Optional: if simulated downsampling) CNN prediction:
   - Applies a CNN model to reconstruct high-resolution images.

6. Aggregating data for QSM:
   - Creates aggregated magnitude image.
   - Creates predicted aggregated local field image.
   - Creates ILR aggregated local field image.
   - Creates a matrix rotation image from the registration.

7. Saving data as MATLAB workspace:
   - Saves the MATLAB workspace for predicted local fields for QSM reconstruction.
   - Saves the MATLAB workspace for ILR local fields for QSM reconstruction.

8. Saving Python variables in a pickle file:
   - Saves Python variables related to the processing steps in a pickle file.

Dependencies:
    - Python 3.7 or higher
    - TensorFlow 2.0 or higher
    - MATLAB Engine for Python
    - NumPy
    - nibabel
    - tqdm
    - warnings
    - pickle
    - scipy

Make sure to have the necessary environment and data paths set correctly before running the script.

"""
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob 
import numpy as np 
import nibabel as nib
import tensorflow as tf
import warnings
import pickle
from pytictoc import TicToc

# assumes SAMO-QSM is saved in python path
from src.qsm_preprocessing import perform_bias_field_correction, perform_phase_unwrapping_and_background_removal, perform_echo_combination, normalize_mag_files, create_matlab_workspace, fsl_mask_creation, perform_registration_and_mask_generation, perform_resizing, perform_localfield_registration_and_interpolation
from src.nifti_file_tools import aggregate_and_save_4D_nifti, create_repeated_nifti, get_voxel_size
from CNN.model_application import batch_apply_model
from src.file_manager import verify_files_exist
from src.extract_TE_from_json import extract_echo_times

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Biasfield correct & FSLBet + Echo combine + Phase unwrapping + Background Removal + Interpolating data + Apply model 

if __name__ == "__main__":
    data_folder = '/home/magic-chusj-2/Documents/E2022/DATA/ex_ds_dataset/data/1.05'
    downsample = True

    # Extract files from subfolders
    mag_suffix = 'magnitude'
    phs_suffix = 'phase'
    msk_suffix = 'mask'

    mag_files = sorted(glob.glob(os.path.join(data_folder, f'*/*{mag_suffix}.nii*')))
    e1_files = sorted(glob.glob(os.path.join(data_folder, '*/*e1.nii')))    # First echo magnitude images
    phs_files = sorted(glob.glob(os.path.join(data_folder, f'*/*{phs_suffix}.nii*')))
    msk_files = sorted(glob.glob(os.path.join(data_folder, f'*/*{msk_suffix}.nii*'))) # Change to msk

    print(f'\nProcessing folder : {data_folder}.')

    # Generate mask if no mask found
    if len(msk_files) == 0 : 
        print("No mask file found. Creating a mask for the central image.")
        msk_file = fsl_mask_creation(e1_files[0])
        print(f"The mask has been successfully created and saved as '{msk_file}'.\n")
        msk_files.append(msk_file)

    # First orientation 
    central_path = os.path.dirname(mag_files[0])
    jsn_files = sorted(glob.glob(os.path.join(central_path, '*_e?.json')))
    if not jsn_files:
        # If json_files is empty, allow manual input of echo times (TEs)
        manual_input = input("Enter echo times (TEs) separated by spaces: ")
        TEs = [float(te) for te in manual_input.split()]
    else:
        # Extract the echo times from the JSON files
        TEs = extract_echo_times(jsn_files)

    n = len(mag_files)
    if n != len(phs_files):
        warnings.warn("Mismatched number of phase and magnitude files. The number of files should match the orientations acquired.")
    
    
    folder_idx = -2
    
    t = TicToc()
    t.tic() #Start timer
    transformation_files = []
    print('\n------------- Image Registration - Central Orientation To Complementary  -------------')
    # Registration of central mask to complementary images 
    central_mag = e1_files[0]
    central_msk = msk_files[0]
    generate_mask = True if len(msk_files) == 1 else False
    transformation_files, complementary_msk_files = perform_registration_and_mask_generation(central_mag, central_msk, e1_files, data_folder, generate_mask=generate_mask)
    msk_files.extend(complementary_msk_files)
    
    print('\n------------- N4BiasFieldCorrection -------------')
    ss_mag_files,  _ = perform_bias_field_correction(e1_files, mag_suffix = 'e1', folder_name_idx = folder_idx, msk_creation = False, msk_files = msk_files)
    # ss : stands for skull stripped
    verify_files_exist(ss_mag_files)

    print('\n ---------- Echo phase combination ----------')
    echoCombined_phs_files = perform_echo_combination(phs_files, mag_files, folder_name_idx=folder_idx)

    print('\n ---------- Phase Unwrapping & Background field removal ----------')
    voxelsize = [get_voxel_size(e1_files[0])]*n

    localfield_files = perform_phase_unwrapping_and_background_removal(echoCombined_phs_files, msk_files, voxelsize, TEs, phs_suffix = f'{phs_suffix}_echoCombined')
    verify_files_exist(localfield_files)

    # IMPORTANT : Normalize magnitude images to fit model training set 
    norm_ss_mag = normalize_mag_files([ss_mag_files[0]])
    norm_ss_mag = norm_ss_mag * n

    # To do : Perform downsampling with nib function (upsampling will be done by registration) (!!) 
    if downsample:
        print('\n---------- Downsampling & Interpolation ----------')
        # scale_factor = 2 # Downsampling scale factor
        # simulated_vsz = [x * scale_factor for x in voxelsize[0]]
        simulated_vsz = [1.88125, 1.88125, 1.88]
        print(f'Simulating data with a resolution of {simulated_vsz[0]}mm x {simulated_vsz[1]}mm x {simulated_vsz[2]}.')

        ofiles = [file.replace('.nii','_sim_LR.nii') for file in localfield_files[1:] ]
        ILR_files = perform_resizing(localfield_files[1:], simulated_vsz, output_files=ofiles) 
    
    print('\n------------- Registration of Local Fields -------------')
    rot_matrices = np.zeros((3,3,n))
    rot_matrices[:,:,0] = np.identity(3)

    central_mag = norm_ss_mag[0]
    fmask = msk_files[0]

    if downsample:
        imgs_to_be_reg = [localfield_files[1:], ILR_files]
    else: 
        imgs_to_be_reg = [localfield_files[1:]]
    
    ofile_reg_locafields = []
    for list_imgs in imgs_to_be_reg:
        reg_localfields, rot_matrices = perform_localfield_registration_and_interpolation(transformation_files, list_imgs, central_mag, data_folder)
        ofile_reg_locafields.extend(reg_localfields)
    
    if downsample: 
        localfield_registered_files = ofile_reg_locafields[:(n-1)]
        ILR_localf_reg_files = ofile_reg_locafields[(n-1):]
    else: 
        localfield_registered_files  = ofile_reg_locafields
    
    if downsample:
        # Apply model prediction to simulated downsampled image
        print('\n------------- Applying CNN model -------------')
        # Load model
        model_file = 'CNN/trained_model.h5' # Relative path to super-resolution trained model
        model = tf.keras.models.load_model(model_file)

        # Apply model prediction 
        print(f'Reconstruction of predicted images by averaging.')
        localfield_pred_files = batch_apply_model(model, ILR_localf_reg_files, norm_ss_mag[1:])
        verify_files_exist(localfield_pred_files)

    print('\n------------- Aggregating data for QSM -------------')
    # Create aggregated magnitude image
    fmag4D = os.path.join(data_folder, 'agg_mag.nii')
    repeated_mag_4d = create_repeated_nifti(ss_mag_files[0], n,fmag4D)

    # Create localfield image
    f_localfield4D = os.path.join(data_folder, 'agg_localfield.nii')
    localfield_registered_files.insert(0, localfield_files[0])
    agg_localfield = aggregate_and_save_4D_nifti(localfield_registered_files, f_localfield4D)

    if downsample: 
        # Create predicted aggragated localfield image
        fpred_localfield4d = os.path.join(data_folder, 'agg_localfield_pred.nii')
        localfield_pred_files.insert(0, localfield_files[0]) # Replace first img file with HR locafield (high res central image)
        agg_localfield_pred = aggregate_and_save_4D_nifti(localfield_pred_files, fpred_localfield4d)

        # Create ILR aggragated localfield image
        fILR_localfield4d = os.path.join(data_folder, 'agg_localfield_ILR.nii')
        ILR_localf_reg_files.insert(0, localfield_files[0]) # Replace first img file with HR locafield (high res central image)
        agg_localfield_ILR = aggregate_and_save_4D_nifti(ILR_localf_reg_files, fILR_localfield4d)

    # Create matrix rotation image (from registration)
    rot_ofile = os.path.join(data_folder, 'rot_all.nii')
    rot_nii = nib.Nifti1Image(rot_matrices, np.eye(4))
    nib.save(rot_nii, rot_ofile)

    # Load mask from mask files 
    msk_nii = nib.load(msk_files[0]) # Mask corresponds to the central position mask
    msk = msk_nii.get_fdata().astype(bool)
    
    print('\n------------- Saving data as matlab workspace -------------')
    # Create matlab workspace for pred localfields
    f_workspace = create_matlab_workspace(data_folder, repeated_mag_4d, msk, agg_localfield, rot_matrices, msk_files[0], 'localfield_workspace.mat', TEs, voxelsize[0])
    print(f"File '{f_workspace}' MATLAB workspace was created")
    
    if downsample:
        f_workspace = create_matlab_workspace(data_folder, repeated_mag_4d, msk, agg_localfield_pred, rot_matrices, msk_files[0], 'localfield_pred_workspace.mat', TEs, voxelsize[0])
        print(f"File '{f_workspace}' MATLAB workspace was created.")
        
        f_workspace = create_matlab_workspace(data_folder, repeated_mag_4d, msk, agg_localfield_ILR, rot_matrices, msk_files[0], 'localfield_ILR_workspace.mat', TEs, voxelsize[0])
        print(f"File '{f_workspace}' MATLAB workspace was created.")
    

    # Save python variables in pickle file 
    pickle_file = os.path.join(data_folder,'pyton_workspace.pkl')
    pickle_data = { "data_folder": data_folder, "mag_files": mag_files, "e1_files": e1_files, "phs_files": phs_files, 
        "msk_files": msk_files, "ss_mag_files": ss_mag_files, "echoCombined_phs_files": echoCombined_phs_files, 
            "voxelsize": voxelsize, "TEs": TEs, "localfield_files": localfield_files, "norm_ss_mag": norm_ss_mag, 
                    "fmag4D": fmag4D, "agg_mag": repeated_mag_4d, "rot_ofile": rot_ofile, "rot_img": rot_matrices, "nifti_file": msk_files[0]}
    if downsample: 
        pickle_data.update({"ILR_files": ILR_files, "localfield_pred_files": localfield_pred_files, "agg_localfield_pred": agg_localfield_pred,})
    with open(pickle_file, "wb") as file:
        pickle.dump(pickle_data, file)

    elapsed_seconds = t.tocvalue()
    print(f"Total elapsed time: {elapsed_seconds/60:.2f} minutes.")
 