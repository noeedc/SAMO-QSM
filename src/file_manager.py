"""
File: file_manager.py
Description: This script provides functions for various file operations, including file existence verification,
             copying files based on wildcard patterns, renaming files, and extracting subfolders.
Author: Noée Ducros-Chabot

Dependencies:
- os
- warnings
- shutil
- glob

Functions:
1. check_file_path(path): Verify if a file exists at the specified path.
2. verify_files_exist(file_list): Verify if files exist in a list of file paths.
3. copy_files_with_wildcard(source_dir, wildcard_pattern, destination_dir): Copy files matching a wildcard pattern.
4. replace_string_in_filenames(folder_path, old_string, new_string): Replace a specific string in filenames within a folder and its subdirectories.
5. copy_files_to_folders(file_list, new_names): Copy a list of files into new folders with specified names.
6. get_subfolders(folder_path): Extract subfolders from a given folder.
7. remove_repetitions(input_string): Remove repetitions in an input string separated by underscores.
8. rename_files_by_list(file_paths, new_file_names): Rename files based on a list of new filenames.

Usage:
1. Each function serves a specific purpose and can be used independently.
2. Update the script's main block to demonstrate the usage of these functions with your specific use case.
"""

import os 
import warnings
import shutil 
import glob

def check_file_path(path):
    """
    Verify if files exist in a list of file paths.

    Parameters:
        file_list (list of str): List of file paths to be verified.

    Returns:
        list of str: List of file paths that do not exist.
    """
    if not os.path.isfile(path):
        warnings.warn(f"The path '{path}' is not a valid file.", category=UserWarning)
        return False
    return True

def verify_files_exist(file_list):
    """
    Verify if files exist in a list of file paths.

    Parameters:
        file_list (list of str): List of file paths to be verified.

    Returns:
        list of str: List of file paths that do not exist.
    """
    existing_files = []
    missing_files = []

    for file_name in file_list:
        if check_file_path(file_name):
            existing_files.append(file_name)
        else:
            missing_files.append(file_name)

    return missing_files

def copy_files_with_wildcard(source_dir, wildcard_pattern, destination_dir):
    """
    Copies files matching a wildcard pattern from the source directory to a nested folder in the destination directory.

    Parameters:
    source_dir (str): The source directory containing the files to be copied.
    wildcard_pattern (str): The wildcard pattern used to match files in the source directory.
    destination_dir (str): The destination directory where the files will be copied to.

    Returns:
    None
    """
    # Create the nested folder
    nested_folder = os.path.join(destination_dir)
    os.makedirs(nested_folder, exist_ok=True)

    # Copy files matching the wildcard pattern to the nested folder
    file_list = glob.glob(os.path.join(source_dir, wildcard_pattern))
    for file_path in file_list:
        relative_path = os.path.relpath(file_path, source_dir)
        destination_path = os.path.join(nested_folder, relative_path)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy(file_path, destination_path)

def replace_string_in_filenames(folder_path, old_string, new_string):
    """
    Replace a specific string in filenames within a folder and its subdirectories.

    Parameters:
    folder_path (str): The path to the root folder.
    old_string (str): The string to be replaced in filenames.
    new_string (str): The new string to replace the old string.

    Returns:
    None
    """
    # Walk through the folder and its subdirectories
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if old_string in filename:
                old_path = os.path.join(root, filename)
                new_filename = filename.replace(old_string, new_string)
                new_path = os.path.join(root, new_filename)

                # Rename the file with the new filename
                os.rename(old_path, new_path)
                print(f"Renamed {old_path} to {new_path}")


def copy_files_to_folders(file_list, new_names):
    """
    Copy a list of files into new folders with specified names.

    Parameters:
        file_list (list): List of file paths to be copied.
        new_names (list): List of new names for the copied files, corresponding to each file in the file_list.

    Returns:
        None
    """
    if len(file_list) != len(new_names):
        raise ValueError("The number of new names must match the number of files.")
    
    for file_path, new_name in zip(file_list, new_names):
        if not os.path.exists(os.path.dirname(new_name)):
            os.makedirs(os.path.dirname(new_name))
        
        if os.path.exists(new_name):
            warnings.warn(f"File already exists in folder '{new_name}' and will be overwritten.")
        
        shutil.copy(file_path, new_name)


def get_subfolders(folder_path):
    """
    Extracts subfolders from a given folder.

    Parameters:
    - folder_path (str): The path to the folder.

    Returns:
    - subfolders (list): A list of subfolder names.
    """
    # Get the list of all items in the folder
    items = os.listdir(folder_path)

    # Filter out only subfolders
    subfolders = [os.path.join(folder_path, item) for item in items if os.path.isdir(os.path.join(folder_path, item))]

    return subfolders


def remove_repetitions(input_string):
    parts = input_string.split('_')  # Split the input string into parts based on underscores
    result_parts = []

    for part in parts:
        if part not in result_parts:
            result_parts.append(part)

    return '_'.join(result_parts)

def rename_files_by_list(file_paths, new_file_names):
    if len(file_paths) != len(new_file_names):
        print("Error: The number of file paths does not match the number of new file names.")
        return

    for file_path, new_filename in zip(file_paths, new_file_names):
        # Get the directory and current filename from the file path
        directory, current_filename = os.path.split(file_path)

        # Construct the full path for the new file name
        new_path = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(file_path, new_path)



if __name__ == "__main__":
    # Script to copy files with wildcard in new folder :
    # source_dir = "/home/magic-chusj-2/Documents/E2022/AMONI/HBHLNeurosphere_13_10_22"
    # wildcard_patterns = ["*/nifti/*e1.nii*", "*/nifti/*all.nii*", "*/nifti/*all_ph.nii*", "*/inverse_registration/*eMask*.nii*", "*/nifti/*eMask.nii*"]
    # destination_dir = "/home/magic-chusj-2/Documents/E2022/AMONI/HBHLNeurosphere_13_10_22_same_mask"

    # for pattern in wildcard_patterns: 
    #     copy_files_with_wildcard(source_dir, pattern, destination_dir)

    # Script to rename files
    # folder_path = "/home/magic-chusj-2/Documents/E2022/AMONI/HBHLNeurosphere_13_10_22_same_mask"
    # old_string = "eMask.nii"  # Replace this with the string you want to replace
    # new_string = "mask.nii"  # Replace this with the new string
    
    # replace_string_in_filenames(folder_path, old_string, new_string)

    # basepath = '/home/magic-chusj-2/Documents/E2022/downsampled_dataset/1.05mm'
    # mask_files = sorted(glob.glob(os.path.join(basepath, '*/inv_registration/*mask*.nii*')))
    # new_mask_files = [ x.replace('/inv_registration','').replace('mag_ss_mask_registered', 'mask') for x in mask_files]
    # copy_files_to_folders(mask_files, new_mask_files)

    # base_path = '/home/magic-chusj-2/Documents/E2022/AMONI/HBHLNeurosphere_13_10_22_for_train'
    # localfield_files = sorted(glob.glob(os.path.join(base_path, '*/nifti/complementary_to_central_reg/*localfield_registered_cropped.nii*')))
    
    # new_names = []
    # for file in localfield_files: 
    #     new_path = os.path.dirname(file).replace('nifti/complementary_to_central_reg', 'training_data')
    #     basename = os.path.basename(file).replace('_registered_cropped', '')
    #     new_basename = remove_repetitions(basename)
    #     filename = os.path.join(new_path, new_basename)
    #     new_names.append(filename)
    
    # copy_files_to_folders(localfield_files, new_names)

    # Rename files for training cnn
    base_path = '/home/magic-chusj-2/Documents/E2022/CNN-Data(copy)'
    localfield_files = sorted(glob.glob(os.path.join(base_path, 'sub-*/ses-01/anat/', '*mag_ss_cropped_1mm.nii*')))
    new_names = [file.replace('mag_ss_cropped_1mm.nii', 'mag_GRE_ss.nii') for file in localfield_files]
    rename_files_by_list(localfield_files, new_names)