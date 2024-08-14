"""
Script Name: Model Application
Description: This script applies a trained model to input data and generates predictions.
Author: Noée Ducros-Chabot
Created: June 8th 2023

Requirements:
- Python [version]
- TensorFlow [version]
- numpy [version]
- nibabel [version]
- matplotlib [version]
- tqdm [version]

Usage:
1. Set the input parameters and paths in the script:
    - cnn_folder: Path to the CNN results folder.
    - model_file: Path to the trained model file (model.h5).
    - pickle_file: Path to the pickle file containing training variables (training_variables.pkl).
    - log_file: Path to the log file for model parameters (model_params.log).
    - data_folder: Path to the data folder containing input images.
    - ILR_suffix: ILR file suffix pattern (default: '*localfield_ILR.nii.gz').
    - HR_suffix: HR file suffix pattern (default: '*mag_GRE_ss.nii.gz').
    - target_suffix: Target file suffix pattern (default: '*localfield.nii.gz').

2. Run the script:
- Ensure that all the required libraries and dependencies are installed.
- Execute the script using a Python interpreter.
- The script will load the trained model, process the test data, perform predictions, and save the output images.

Note:
- This script assumes a specific folder structure for input data. Adjust the paths and patterns accordingly if your folder structure is different.
- Make sure the necessary data files are present in the specified folders.
- The script will generate predictions for the test data and save the predicted images in the results folder.
- The script also logs model parameters and displays a histogram of the losses during testing.
- Ensure that you have sufficient computational resources to run the script, as it involves processing a large amount of data.

"""

import numpy as np 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import glob 
import nibabel as nib
from tqdm import tqdm
import math
import warnings

from CNN.patchify import generate_3D_sliding_window_patches, reconstruct_image_with_overlap
from CNN.cnn_training_script import get_batch_data

warnings.filterwarnings("ignore", category=RuntimeWarning)

def fill_zeros(array):
    """
    Fill zeros in an array to ensure it has shape (1, 64, 25, 25, 25, 2).

    Args:
        array (ndarray): Input array.

    Returns:
        ndarray: Array with zeros filled to the remaining dimensions.
    """
    target_shape = (1, 64, 25, 25, 25, 2)

    if array.shape == target_shape:
        return array

    filled_array = np.zeros(target_shape, dtype=array.dtype)

    filled_array[:, :array.shape[1], :array.shape[2], :array.shape[3], :array.shape[4], :array.shape[5]] = array

    return filled_array

def apply_model(ILR_file, HR_file, model, patch_size=[25, 25, 25], batch_size=64):
    """
    Apply a trained model to a single pair of ILR and HR images and generate a prediction.

    Args:
        ILR_file (str): Path to the ILR image file.
        HR_file (str): Path to the HR image file.
        model (tf.keras.Model): Trained model to apply.
        patch_size (list, optional): Size of the patches. Defaults to [25, 25, 25].
        batch_size (int, optional): Batch size for prediction. Defaults to 64.

    Returns:
        np.ndarray: Predicted image array.

    Raises:
        IndexError: If no ILR or HR files are found.
    """
    # Prep subject data
    patch_indices, ILR_patches, HR_patches, _ = generate_3D_sliding_window_patches(patch_size, ILR_file, HR_file)
    eval_input_data = np.stack((ILR_patches, HR_patches), axis=-1)

    steps = math.ceil(len(patch_indices) / batch_size)
    pred_patches = np.empty([0] + patch_size)
    for step in range(steps):
        # Get batch data
        intial_size = None
        x_batch, _ = get_batch_data(batch_size, step, eval_input_data)

        # Assures that the input is the right size for the model (predicts zeros)
        if np.shape(x_batch) != (1, 64, 25, 25, 25, 2):
            intial_size = np.shape(x_batch)
            x_batch = fill_zeros(x_batch)

        # Predict on the batch
        pred_patch = model.predict_on_batch(x_batch)

        if intial_size: 
            pred_patch = pred_patch[:, :intial_size[1], :intial_size[2], :intial_size[3], :intial_size[4]]

        pred_patches = np.concatenate((pred_patches, pred_patch[0, :]), axis=0)

    target_nii = nib.load(ILR_file)
    image_shape = target_nii.header.get_data_shape()

    pred_img = reconstruct_image_with_overlap(pred_patches, image_shape, patch_size)
    if np.isnan(pred_img).any():
        pred_img = np.nan_to_num(pred_img, nan=0)

    return pred_img

def batch_apply_model(model, ILR_files, HR_files, patch_size=[25, 25, 25], batch_size=64, ofiles=None):
    """
    Apply a trained model to a folder containing input data and generate a prediction.

    Args:
        model (tf.keras.Model): Trained model to apply.
        ILR_files (str, optional): List of interpolated low resolution nifti volume filepaths.
        HR_files (str, optional): List of high resolution resolution nifti volume filepaths.
        patch_size (list, optional): Size of the patches. Defaults to [25, 25, 25].
        batch_size (int, optional): Batch size for prediction. Defaults to 64.
        ofile (str, optional): Output file path for the predicted image. Defaults to None. If not precised, 'localfield_pred.nii' will be created

    Returns:
        str: Path to the predicted image file.

    Raises:
        IndexError: If no ILR or HR files are found in the input folder.
    """
    # TO DO : Make sure they are the same lenght
    ILR_files = sorted(ILR_files)
    HR_files = sorted(HR_files)
    prediction_files = []

    if len(ILR_files) == 0 or len(HR_files) == 0:
        raise IndexError("No ILR or HR files found in the input folder.")

    set_ofile = (ofiles == None) # Bool to check if output filename needs to be set or is put as input
    progress_bar = tqdm(ILR_files, desc="Applying model for high resolution prediction.", unit="file(s)", ncols=120)
    for i, ILR_file, HR_file in zip(range(len(HR_files)),ILR_files, HR_files):
        progress_bar.set_postfix({"Current File": os.path.basename(ILR_file)})
        pred_img = apply_model(ILR_file, HR_file, model, patch_size, batch_size)

        # Save output image
        target_nii = nib.load(ILR_file)
        pred_nii = nib.Nifti1Image(pred_img, target_nii.affine, header=target_nii.header)

        if set_ofile and 'ILR' in ILR_file:
            ofile = ILR_file.replace('localfield_ILR.nii', 'localfield_pred.nii')
        elif set_ofile and 'ILR' not in ILR_file :
            ofile = ILR_file.replace('.nii', '_pred.nii')
        else:
            ofile = ofiles[i]

        # Create folder if it does not already exist
        folder_name = os.path.dirname(ofile)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        nib.save(pred_nii, ofile)
        prediction_files.append(ofile)
        progress_bar.update(1)
    progress_bar.close()
    return prediction_files
    

if __name__=="__main__":
    # Example usage 1
    # cnn_folder = '/home/magic-chusj-2/Documents/E2022/CNN_results/model_20230614_1704'
    
    # # Load model
    # model_file = os.path.join(cnn_folder, 'model.h5')
    # model = tf.keras.models.load_model(model_file)
    # model.summary()

    # # Load variables from training script
    # pickle_file = os.path.join(cnn_folder,'training_variables.pkl')
    # file = open(pickle_file, 'rb')
    # train_subfolders, patch_size, batch_size, input_shape = pickle.load(file) # could also extract from log file
    # file.close()

    # log_file = os.path.join(cnn_folder, 'model_params.log')
    # logging.basicConfig(filename=os.path.join(cnn_folder, 'model_params.log'), filemode='a', format='%(message)s', level=logging.INFO) # Test this append function

    # data_folder = '/home/magic-chusj-2/Documents/E2022/CNN-Data'
    # subfolders = glob.glob(os.path.join(data_folder, 'sub-*/ses-01/anat/'))
    
    # Testing the model
    # n_test = 5
    # test_subfolders = np.random.choice([s for s in subfolders if s not in train_subfolders], n_test)
    
    # print('------------ Loading & Patching Test Data ------------')
    # #Create a tqdm progress bar
    # progress_bar = tqdm(test_subfolders, desc="Patching testing data", unit="file(s)",  ncols=120)
    # logging.info('Testing Set: Patches generated with sliding window with default stride (patch_size/2)')

    # ILR_suffix = '*localfield_ILR.nii.gz' # local field interpolated low resolution image
    # HR_suffix = '*mag_GRE_ss.nii.gz' # skull stripped magnitude image (bias field corrected)
    # target_suffix = '*localfield.nii.gz' # localfield in Hz

    # test_ILR_patches = np.empty([0] + patch_size)
    # test_HR_patches = np.empty([0] + patch_size)
    # test_target_patches = np.empty([0] + patch_size)
    # for test_subfolder in test_subfolders: 
    #     folder_name = test_subfolder.split('/')[6]
    #     progress_bar.set_postfix({"Current Folder": folder_name})
    #     progress_bar.update(1)

    #     ILR_file = glob.glob(os.path.join(test_subfolder, ILR_suffix))[0]
    #     HR_file = glob.glob(os.path.join(test_subfolder, HR_suffix))[0]
    #     target_file = glob.glob(os.path.join(test_subfolder, target_suffix))[0]
        
    #     patch_indices, ILR_patches, HR_patches, ground_truth_patches = generate_3D_sliding_window_patches(ILR_file, HR_file, target_file, patch_size)

    #     test_ILR_patches = np.concatenate((test_ILR_patches, ILR_patches), axis=0)
    #     test_HR_patches = np.concatenate((test_HR_patches, HR_patches), axis=0)
    #     test_target_patches = np.concatenate((test_target_patches, ground_truth_patches), axis=0)
    # progress_bar.close()
    # logging.info(f'Testing image yield {len(patch_indices)} patches per subject.')
    # test_input_data = np.stack((test_ILR_patches, test_HR_patches), axis=-1)

    # print('------------ Test Model ------------')
    # num_samples = test_input_data.shape[0]
    # steps_per_epoch = num_samples // batch_size
    # results = []
    # for step in range(steps_per_epoch):
    #         # Get batch data
    #         x_batch, y_batch = get_batch_data(test_input_data, test_target_patches, step=step, batch_size=batch_size)

    #         # Train the model on the batch
    #         result = model.test_on_batch(x_batch, y_batch)
    #         results.append(result[0])

    # # Plotting histogram of the losses 
    # plt.hist(results, bins=10, alpha=0.7, color='blue', edgecolor='black')
    # plt.title('Histogram of losses (Mean squared error)')
    # plt.ylabel('Frequency (batch of 64)')
    # plt.xlabel('Mean squared error')
    # plt.show()

    # print('------------ Appy to all subjects not in training set ------------')
    # subjects_not_in_training_set = [subject for subject in subfolders if subject not in train_subfolders]
    # subjects_not_in_training_set = sorted(subjects_not_in_training_set)

    # logging.info('\n--------------\nModel Predict & Image Reconstruction\n--------------')
    # logging.info(f'Reconstruction method: averaging')

    # ILR_suffix='*localfield_ILR.nii*'
    # HR_suffix='*mag_*ss.nii*' # To be verified

    # progress_bar = tqdm(range(len(subjects_not_in_training_set)), desc="Applying model prediction", unit="file(s)", ncols=120)
    # for subfolder in subjects_not_in_training_set: 
    #         folder_name = subfolder.split('/')[6]
    #         progress_bar.set_postfix({"Current Folder": folder_name})
    #         progress_bar.update(1)
            
    #         ofile = os.path.join(cnn_folder, folder_name, f'{folder_name}_localfield_pred.nii.gz')
            

    #         ILR_files = glob.glob(os.path.join(subfolder, ILR_suffix))
    #         HR_files = glob.glob(os.path.join(subfolder, HR_suffix))
    #         batch_apply_model(model, ILR_files, HR_files,  ofile = ofile) # Could maybe use the single apply

    # Example usage 2 
    data_path = '/home/magic-chusj-2/Documents/E2022/downsampled_dataset_nCNN/1.9mm'
    ILR_files = sorted(glob.glob(os.path.join(data_path, '*/complementary_to_central_reg/*localfield_registered.nii')))
    skull_stripped_mag = '/home/magic-chusj-2/Documents/E2022/downsampled_dataset_nCNN/1.9mm/central/dicom_gre_IPAT2_pPF_sPF_3D_6echoes_1.05_20230829132927_2_mag_ss_norm.nii.gz'
    HR_files = [skull_stripped_mag]*len(ILR_files)
    
    cnn_folder = '/home/magic-chusj-2/Documents/E2022/CNN_results/model_20230913_1423'
    model_file = os.path.join(cnn_folder, 'model.h5')
    model = tf.keras.models.load_model(model_file)
    
    outpath = os.path.join('/home/magic-chusj-2/Documents/E2022/downsampled_dataset_nCNN', os.path.basename(cnn_folder))
    ofiles = [ os.path.join(outpath, os.path.basename(file).replace('ILR', 'pred'))for file in ILR_files]
    batch_apply_model(model, ILR_files, HR_files, patch_size=[25, 25, 25], batch_size=64, ofiles=ofiles)