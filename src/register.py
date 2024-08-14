#!/usr/bin/env python
# coding: utf-8

"""
File: registration_utils.py
Author: [Your Name]
Description: This script provides utility functions for image registration using ANTs.

Functions : 
     - ants_rigid_registration
     - ants_affine_registration
     - ants_apply_single_transform
     - ants_syn_registration


"""

import os
import glob
import math
import numpy as np
import numpy.linalg as npl
import nibabel as nib
import numpy as np
from pytictoc import TicToc
from dipy.core.geometry import decompose_matrix
from dipy.align.imaffine import AffineMap
from dipy.align import (affine_registration, center_of_mass, translation,
                        rigid)
from dipy.align.metrics import *
from skimage.transform import resize

def ants_rigid_registration(outpath, ofile, fixedImage, movingImage, verbose=False, interpolate = 'Linear'):
     """
     Perform rigid registration using ANTs.

     Parameters:
     - outpath (str): Output path for the registration results.
     - ofile (str): Output filename.
     - fixedImage (str): Path to the fixed image.
     - movingImage (str): Path to the moving image.
     - verbose (bool): Verbosity flag.
     - interpolate (str): Interpolation method.

     Returns:
     None
     """
     verbose = 1 if verbose else 0
     allowed_interpolators = [
          'Linear',
          'NearestNeighbor',
          'MultiLabel',
          'Gaussian',
          'BSpline',
          'CosineWindowedSinc',
          'WelchWindowedSinc',
          'HammingWindowedSinc',
          'LanczosWindowedSinc',
          'GenericLabel'
     ]

     if interpolate not in allowed_interpolators:
          raise ValueError(f"Invalid 'interpolate' option: {interpolate}. Allowed options are: {', '.join(allowed_interpolators)}")
     
     os.system(f'antsRegistration --dimensionality 3 --float 0 \
                    --output [{outpath}, {os.path.join(outpath, ofile)}] \
                    --interpolation {interpolate} \
                    --winsorize-image-intensities [0.005,0.995] \
                    --use-histogram-matching 1 \
                    --initial-moving-transform [{fixedImage},{movingImage},0] \
                    --transform Rigid[0.1] \
                    --metric MI[{fixedImage},{movingImage},1,32,Regular,0.25] \
                    --convergence [1000x500x250x100,1e-6,10] \
                    --shrink-factors 8x4x2x1 \
                    --smoothing-sigmas 3x2x1x0vox\
                    --write-composite-transform 1 \
                    --verbose {verbose}' )
     return

def ants_affine_registration(outpath, ofile, fixedImage, movingImage, verbose=False):
     """
     Perform affine registration using ANTs.

     Parameters:
     - outpath (str): Output path for the registration results.
     - ofile (str): Output filename.
     - fixedImage (str): Path to the fixed image.
     - movingImage (str): Path to the moving image.
     - verbose (bool): Verbosity flag.

     Returns:
     None
     """
     verbose = 1 if verbose else 0

     os.system(f'antsRegistration --dimensionality 3 --float 0 \
                    --output [{outpath}, {os.path.join(outpath, ofile)}] \
                    --interpolation Linear \
                    --winsorize-image-intensities [0.005,0.995] \
                    --use-histogram-matching 1 \
                    --initial-moving-transform [{fixedImage},{movingImage},0] \
                    --transform Rigid[0.1] \
                    --metric MI[{fixedImage},{movingImage},1,32,Regular,0.25] \
                    --convergence [1000x500x250x100,1e-6,10] \
                    --shrink-factors 8x4x2x1 \
                    --smoothing-sigmas 3x2x1x0vox\
                    --transform Affine[0.1] \
                    --metric MI[{fixedImage},{movingImage},1,32,Regular,0.25] \
                    --convergence [1000x500x250x100,1e-6,10] \
                    --shrink-factors 8x4x2x1 \
                    --smoothing-sigmas 3x2x1x0vox \
                    --write-composite-transform 1 \
                    --verbose {verbose}' )
     return

def ants_apply_single_transform(transform_file, input_image, ref_image, output_file, interpolate = 'Linear', verbose = False):
     """
    Apply a single ANTs transformation to an image.

    Parameters:
    - transform_file (str): Path to the transformation file.
    - input_image (str): Path to the input image.
    - ref_image (str): Path to the reference image.
    - output_file (str): Output filename.
    - interpolate (str): Interpolation method.
    - verbose (bool): Verbosity flag.

    Returns:
    None
    """
     verbose = 1 if verbose else 0

     allowed_interpolators = [
          'Linear',
          'NearestNeighbor',
          'MultiLabel',
          'Gaussian',
          'BSpline',
          'CosineWindowedSinc',
          'WelchWindowedSinc',
          'HammingWindowedSinc',
          'LanczosWindowedSinc',
          'GenericLabel'
     ]

     if interpolate not in allowed_interpolators:
          raise ValueError(f"Invalid 'interpolate' option: {interpolate}. Allowed options are: {', '.join(allowed_interpolators)}")
          
     # --dimensionality {dim} \
     os.system(f'antsApplyTransforms \
               --input {input_image} \
               --reference-image {ref_image} \
               --output {output_file} \
               --transform {transform_file} \
               --interpolation {interpolate} \
               --verbose {verbose}\
               ')
     return

def ants_syn_registration(outpath, ofile, fixedImage, movingImage, verbose=False):
     """
     Perform symmetric normalization (SyN) registration using ANTs.

     Parameters:
     - outpath (str): Output path for the registration results.
     - ofile (str): Output filename.
     - fixedImage (str): Path to the fixed image.
     - movingImage (str): Path to the moving image.
     - verbose (bool): Verbosity flag.

     Returns:
     None
     """
     verbose = 1 if verbose else 0
     os.system(f'antsRegistration --dimensionality 3 --float 0 \
               --output [{outpath}, {os.path.join(outpath, ofile)}] \
               --interpolation Linear \
               --winsorize-image-intensities [0.005,0.995] \
               --use-histogram-matching 1 \
               --initial-moving-transform [{fixedImage},{movingImage},0] \
               --transform Rigid[0.1] \
               --metric MI[{fixedImage},{movingImage},1,32,Regular,0.25] \
               --convergence [1000x500x250x100,1e-6,10] \
               --shrink-factors 8x4x2x1 \
               --smoothing-sigmas 3x2x1x0vox\
               --transform Affine[0.1] \
               --metric MI[{fixedImage},{movingImage},1,32,Regular,0.25] \
               --convergence [1000x500x250x100,1e-6,10] \
               --shrink-factors 8x4x2x1 \
               --smoothing-sigmas 3x2x1x0vox \
               --transform SyN[0.1,3,0] \
               --metric CC[{fixedImage},{movingImage},1,4] \
               --convergence [100x70x50x20,1e-6,10] \
               --shrink-factors 8x4x2x1 \
               --smoothing-sigmas 3x2x1x0vox \
               --write-composite-transform 1\
               --verbose {verbose}')
     return

if __name__ == "__main__":
     # Example usage
     data_folder = '/home/magic-chusj-2/Documents/E2022/downsampled_dataset_nCNN/1.9mm'
     fixed_files = sorted(glob.glob(os.path.join(data_folder, '*/complementary_to_central_reg/*localfield_registered.nii*')))
     moving_files = sorted(glob.glob(os.path.join(data_folder, '*/complementary_to_central_reg/*localfield_registered_pred.nii*')))

     for fixedImg, movingImg in zip(fixed_files, moving_files):
          outputpath = os.path.join(os.path.dirname(movingImg), 'localfield_reg')+'/'
          ofile = movingImg.replace('.nii', '_reg.nii')
          ants_rigid_registration(outputpath, ofile, fixedImg, movingImg, interpolate = 'BSpline')