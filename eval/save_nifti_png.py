"""
Script: NIfTI Slicing and Saving

Description:
This script slices NIfTI images and saves the slices as PNG files. It provides flexibility
for slicing multi-orientation and single orientation datasets, with an option to add zoom
to the saved PNG files.

Author: Noée Ducros-Chabot
Date: Decembre 14th 2023

Dependencies:
- numpy
- nibabel
- matplotlib
- os
- glob

Usage:
1. Ensure the required libraries are installed: numpy, nibabel, matplotlib.
   You can install them using: pip install numpy nibabel matplotlib

2. Update the 'basefolder' and 'outputfolder' variables with the appropriate paths.

3. Set the 'zoom' variable to True or False based on whether you want to include zoomed slices.

4. Run the script to slice NIfTI images and save the slices as PNG files.

Note: This script assumes a specific structure of the input NIfTI files and requires the specified dependencies.

"""
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os 
import glob

def slice_nifti_and_save(nifti_filename, x_range, y_range, z_coord, output_png_filename, contrast_range = [-0.1,0.15] ):
    # Load the NIfTI image
    nifti_img = nib.load(nifti_filename)
    img_data = nifti_img.get_fdata()

    # Ensure the provided coordinates are within valid ranges
    x_min, x_max = x_range
    y_range = slice(max(0, y_range[0]), min(img_data.shape[1], y_range[1]))
    z_coord = max(0, min(img_data.shape[2] - 1, z_coord))

    # Extract the desired slice
    slice_data = img_data[x_min:x_max, y_range, z_coord]

    if contrast_range:
        vmin, vmax = contrast_range
        plt.imshow(np.rot90(slice_data), cmap='gray', vmin=vmin, vmax=vmax)
    else:
        plt.imshow(np.rot90(slice_data), cmap='gray')

    # Plot and save the slice as a PNG image
    plt.axis('off')
    plt.savefig(output_png_filename, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

# Example usage
if __name__ == "__main__":
    basefolder = "/home/magic-chusj-2/Documents/E2022/downsampled_dataset_nCNN/1.9mm/results"
    outputfolder = "/home/magic-chusj-2/Documents/E2022/downsampled_dataset_nCNN/1.9mm/results/figures/perf_fig"
    zoom = False
    
    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    
    # Load files for multi-orientation approach
    folders = ["ILR_localfields/w0.5_tkd0.20", "pred_localfields/w0.5_tkd0.20", 'HR_localfields/w0.0_tkd0.00', 'singleOri/central']
    suffixs = ["*ori3_CFR.nii*", "*ori4_CELR.nii*", "*ori5*.nii*", '*Star-QSM*.nii*' ]
    qsm_files = [sorted(glob.glob(os.path.join(basefolder, folder, suffix))) for suffix in suffixs for folder in folders]

    # Single Orientation data 
    qsm_files = [item for sublist in qsm_files for item in sublist]

    for file in qsm_files:
        basename = os.path.basename(file).replace('.nii','.png').replace('.gz', '')
        
        if 'pred_localfields' in file:
            basename = basename.replace('sCosmos_', '')
            basename = basename.replace('.png', '_SAMO-QSM.png')
        elif 'ILR_localfields' in file:
            basename = basename.replace('sCosmos_', '')
            basename = basename.replace('.png', '_ILR-mCOSMOS.png')
        elif 'HR_locafields' in file:
            basename = basename.replace('sCosmos_', '')
            basename = basename.replace('.png', '_COSMOS.png')

        # Move diff into prefix if it exist 
        if "_diff" in basename:
            basename = basename.replace('_diff', '')
            basename = 'diff_'+basename
        
        if zoom :
            # Add zoom to filename
            basename = basename.replace('.png','_zoom.png')
        
            x_range = [64, 230]  
            y_range = [12, 172]
        else : 
            x_range = [110, 184] 
            y_range = [78, 124]
    
        z_coord = 73
        output_png_filename = os.path.join(outputfolder, basename)

        slice_nifti_and_save(file, x_range, y_range, z_coord, output_png_filename, contrast_range = [-0.10,0.15])
