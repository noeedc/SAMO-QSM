"""
CNN Training Script
Author: Noée Ducros-Chabot 
Created : 6th June 2023

Description:
This script performs training of a convolutional neural network (CNN) model for a specific task. It prepares the input data, generates patches, creates the CNN model, and trains the model on the provided data. The trained model is saved to disk along with other relevant variables.

Libraries Used:
- numpy (as np)
- os
- tensorflow (as tf)
- datetime
- glob
- logging
- tqdm
- pickle
- pytictoc
- patchify
- model (your own module)

Usage:
1. Set the base result folder, basefolder (containing the data) and the number of training subjects.
2. Execute the script using the Python interpreter.
3. The script will generate training data, create and train the CNN model, and save the trained model and variables.

Note:
- Make sure to install all the required libraries and dependencies before running the script.
- Adjust the script parameters and configurations according to your specific task and dataset.
- Customize the model architecture and training settings in the "create_model" function of the "model" module.
- Modify the logging settings and file paths as needed.
- Ensure that the necessary input data files and directories are correctly specified.

"""
import numpy as np 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import logging
from tqdm import tqdm
import pickle 
from datetime import datetime
import datetime
from pytictoc import TicToc

from CNN.patchify import generate_3D_rand_patches
from CNN.model import create_model


def get_batch_data(batch_size, step, input_data, target_data = None):
    # New implementation is untested
    # Get the start and end indices for the current batch 
    start_index = step * batch_size
    end_index = (step + 1) * batch_size

    # Get the batch data
    x_batch = input_data[start_index:end_index]
    if target_data is not None : 
        y_batch = target_data[start_index:end_index]
    else :
        y_batch = None

    # Add an extra dimension to match batch size
    x_batch = np.expand_dims(x_batch, axis=0)
    if target_data is not None :y_batch = np.expand_dims(y_batch, axis=0)

    return x_batch, y_batch # Return None as y_batch if target_data not specified

def split_number(n):
    """
    Splits a number into folds, ensuring the most even distribution possible.
    Each fold cannot exceed 20.

    Args:
        n (int): The number to be split into folds.

    Returns:
        list: A list of integers representing the folds.

    """
    n_folds = int(np.ceil(n/20))
    float_array = [n/n_folds]*n_folds

    folds = [int(np.floor(num)) for num in float_array]
    remainder = round(sum(float_array)) - sum(folds)
    sorted_indices = sorted(range(len(float_array)), key=lambda x: float_array[x] - folds[x], reverse=True)

    for i in range(int(remainder)):
        folds[sorted_indices[i]] += 1

    return folds


def choose_random_folders(folders, num_folders):
    """
    Chooses a random set of folders from the given list based on the provided number of folders.

    Args:
        folders (list): The list of folder names.
        num_folders (list): The array of numbers representing the number of folders to choose.

    Returns:
        list: A list of arrays, where each inner array contains randomly chosen folder names.

    Raises:
        ValueError: If the number of folders requested is greater than the available folders.

    """
    if sum(num_folders) > len(folders):
        raise ValueError("Number of folders requested exceeds the available folders.")

    chosen_folders = []
    indexes = list(range(len(folders)))

    for num in num_folders:
        if num > len(indexes):
            raise ValueError("Number of folders requested exceeds the remaining available folders.")

        random_indexes = np.random.choice(indexes, size=num, replace=False)
        chosen_folders.append([folders[i] for i in random_indexes])
        indexes = [i for i in indexes if i not in random_indexes]

    return chosen_folders


if __name__=="__main__":
    base_result_folder ='/home/magic-chusj-2/Documents/E2022/CNN_results'
    
    timestamp = datetime.datetime.now()
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M")
    model_folder = os.path.join(base_result_folder, f'model_{timestamp_str}')
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    logging.basicConfig(filename=os.path.join(model_folder, 'model_params.log'), filemode='w', format='%(message)s', level=logging.INFO)
    logging.info('Prepping input data for CNN')
    logging.info(f'Running script: {__file__}')
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logging.info(f'Timestamp: {current_datetime}')

    basefolder = '/home/magic-chusj-2/Documents/E2022/CNN-Data(copy)'
    subfolders = glob.glob(os.path.join(basefolder, 'sub-*/ses-01/anat/'))

    # Select training set
    print('\n------------ Training Data ------------')
    n_train = 20
    n_training_folds = split_number(n_train) # [Number of subjects] x training folds
    training_folds = choose_random_folders(subfolders, n_training_folds) # [ List of path to training subjects ] x number of training folds
    all_training_folders =  sorted([item for sublist in training_folds for item in sublist])
    
    logging.info('\n---------------------\nTraining set\n---------------------')
    logging.info(f'Number of training subjects: {n_train}')
    print(f'Number of training subjects: {n_train}')
    logging.info(f'Number of training folds: {len(n_training_folds)}')
    logging.info(f'Number of subjects per training folds: {n_training_folds}')
    logging.info(f'Training subjects: {all_training_folders}')

    logging.info('\n------------\nPatching\n------------')
    patch_size = [25, 25, 25]
    n_patch = 3200 # Number of patch per subjects

    print(f'Patch size: {patch_size}')
    logging.info(f'Patch size: {patch_size}')
    logging.info(f'Number of patches per subject: {n_patch}')
    logging.info('Training Set: Patches generated randomly from 3D image')

    # Creating CNN model
    print('\n------------ Creating the CNN Model ------------')
    # Set the batch size and number of epochs
    batch_size = 64
    input_shape = tuple([batch_size]+ list(patch_size)+[2]) # Two channels one containing the interpolated low resolution image (ILR) and one containing the high resolution magnitude image

    model = create_model(input_shape, num_blocks = 10, num_filters = 64, filter_size = [3, 3, 3])
    model.summary() # This might give me error input shape does not seem proper in model summary

    # Comment: Could also use fit_generator
    print('\n------------ Training the Model ------------')
    # Saved training variables
    pickle_file = os.path.join(model_folder, 'training_variables.pkl')
    with open(pickle_file, 'wb') as file:
        pickle.dump([all_training_folders, patch_size, batch_size, input_shape], file) # Change post_cnn_training code accordingly

    # Setting traing parameters
    num_epochs = 20

    logging.info('\n--------------\nCNN parameters\n--------------')
    logging.info(f'Batch size: {batch_size}')
    logging.info(f'Input shape: {input_shape}')
    logging.info(f'Number of epochs: {num_epochs}')

    # File suffix for loading training data
    ILR_suffix = '*localfield_ILR.nii.gz' # local field interpolated low resolution image
    HR_suffix = '*mag_GRE_ss.nii.gz' # skull stripped magnitude image (bias field corrected)
    target_suffix = '*localfield.nii.gz' # localfield in Hz
    
    tot_number_samples = []
    for i, train_subfolders in enumerate(training_folds) : 
        # Create a tqdm progress bar
        print(f'Processing {i+1} training fold...')
        progress_bar = tqdm(train_subfolders, desc="Patching training data", unit="file(s)",  ncols=120)

        # Loading training patches
        train_ILR_patches = np.empty([0] + patch_size)
        train_HR_patches = np.empty([0] + patch_size)
        train_target_patches = np.empty([0] + patch_size)
        for train_subfolder in train_subfolders:
            folder_name = train_subfolder.split('/')[6]
            progress_bar.set_postfix({"Current Folder": folder_name})
            progress_bar.update(1)

            ILR_file = glob.glob(os.path.join(train_subfolder, ILR_suffix))[0]
            HR_file = glob.glob(os.path.join(train_subfolder, HR_suffix))[0]
            target_file = glob.glob(os.path.join(train_subfolder, target_suffix))[0]

            patch_indices, ILR_patches, HR_patches, ground_truth_patches = generate_3D_rand_patches(ILR_file, HR_file, target_file, patch_size, n_patch)

            train_ILR_patches = np.concatenate((train_ILR_patches, ILR_patches), axis=0)
            train_HR_patches = np.concatenate((train_HR_patches, HR_patches), axis=0)
            train_target_patches = np.concatenate((train_target_patches, ground_truth_patches), axis=0)
        progress_bar.close()
        print('\n')

        del patch_indices, ILR_patches, HR_patches, ground_truth_patches # save memory space 

        # Concatenate ILR and HR into a new channel to form input data
        input_data = np.stack((train_ILR_patches, train_HR_patches), axis=-1)
        
        del train_ILR_patches, train_HR_patches

        # Shuffle the training data 
        num_samples = input_data.shape[0]
        tot_number_samples.append(num_samples)
        shuffle_indices = np.arange(num_samples)
        np.random.shuffle(shuffle_indices)

        shuffled_input_data = np.take(input_data, shuffle_indices, axis=0) # np.take() saves memory
        shuffled_train_target_patches = np.take(train_target_patches, shuffle_indices, axis=0)
        del shuffle_indices

        if i == 0 :
            # Start the timer for the training 
            timer = TicToc() #tic only if first training fold
            timer.tic()

        # Training loop
        steps_per_epoch = num_samples // batch_size  # Calculats the number of steps per epoch based on the batch size
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Create a tqdm progress bar for each epoch
            progress_bar = tqdm(range(steps_per_epoch), desc="Training", unit="batch(s)", ncols=120)

            for step in range(steps_per_epoch):
                # Get batch data 
                x_batch, y_batch = get_batch_data(batch_size, step, shuffled_input_data, shuffled_train_target_patches)
                
                # Train the model on the batch
                loss,_ = model.train_on_batch(x_batch, y_batch)

                # Update the progress bar description
                progress_bar.set_postfix({"Training Loss": loss})
                progress_bar.update(1)

            # Close the progress bar for the epoch
            progress_bar.close()
        del shuffled_input_data, shuffled_train_target_patches, input_data, train_target_patches # Clear training fold data
    print('Training finished')

    logging.info(f'Number of samples: {np.sum(tot_number_samples)}')
    logging.info(f'Steps per epochs: {steps_per_epoch}')
    logging.info('Number of blocks: 10 (default)')
    logging.info('Number of convolution filter: 64 (default)')
    logging.info('Size of convolution filter: [3, 3, 3] (default)')
    logging.info('Input: ILR localfield in first channel, magnitude HR image in second channel')
    logging.info('Network Architecture: 10 Conv&Relu blocks + Conv with one filter + Skip connection between ILR phase and ouput')

    # Get the elapsed time in seconds
    timer.toc()
    elapsed_time = timer.tocvalue()
    time_delta = datetime.timedelta(seconds=elapsed_time)
    formatted_time = str(time_delta)
    print(f'Training time: {formatted_time}')
    logging.info(f'Training time: {formatted_time}')
    logging.info(f'Final training loss: {loss}')

    # Save model
    logging.info('\n--------------\nSaving model & variables\n--------------')
    logging.info(f'Model saved at file: {os.path.join(model_folder,"model.h5")}')
    logging.info(f'Pickle saved variables saves at file: {pickle_file}')
    model.save(os.path.join(model_folder,"model.h5"))
    print("Saved model to disk")

