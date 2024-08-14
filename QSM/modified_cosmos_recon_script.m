% -------------------------------------------------------------------------
% Modified COSMOS-QSM Reconstruction Script
% -------------------------------------------------------------------------
%
% Purpose :
% Script to perform quantitative susceptibility mapping (QSM) using COSMOS algorithm on magnetic resonance imaging (MRI) data.
% This script processes a set of 3D MRI images, and returns a set of 3D QSM images using the COSMOS algorithm.
% The input MRI data should be in the .mat format, and include the following variables: kernel, msk, vsz, and N.
% The COSMOS algorithm uses a dipole inversion technique to estimate magnetic susceptibility from the MRI data.
% This script includes options to perform single solution dipole inversions, with varying threshold values and regularization terms.
% The output QSM images are saved in the .nii format.
%
% Usage:
%   1. Set the path to the COSMOS-QSM directory.
%   2. Specify the input .mat file containing workspace information.
%   3. Define dipole inversion options and orientations.
%   4. Run the script to process the COSMOS-QSM data, generating
%      susceptibility maps for different orientations and parameters.
% 
% Notes:
%   - This script assumes the existence of relevant functions such as 
%     cosmos and createKernel.
%   - Adjustments may be needed based on the specific data and requirements.
%
% Example Usage:
%   modified_cosmos_recon_script;
%
% -------------------------------------------------------------------------
% Author: Noée Ducros-Chabot
% Date: 24/11/2023
% -------------------------------------------------------------------------


%% Inputs 

% Add the directory path to MATLAB path
path = '/home/magic-chusj-2/Documents/E2022/SAMO-QSM/QSM'; % directory path
addpath(genpath(path)); % add directory to MATLAB path

% Load input data from .mat file
input_workspace = '/home/magic-chusj-2/Documents/E2022/DATA/ex_ds_dataset/data/1.05/localfield_workspace.mat';

% Define options for dipole inversion and combinations of orientations
appliedOri = ["ori5", "ori4", "ori3"]; % ["ori5", "ori4", "ori3"]
TKDs = 0.2 ; % Threshold values for TKD truncation. Default 0.2. (Ex. 0:0.05:0.25).
weights = 0.5 ; % Weight assigned to the central image. The remaining weight will be distributed among complementary orientations. 
                        % Must be within the range [0, 1]. Default = 0.5.

%% Load workspace
load(input_workspace);
info = niftiinfo(nifti_file);

% Erode mask 
se = strel('sphere', 3);  % Create a 3x3x3 spherical structuring element (neighborhood) for erosion
msk = imerode(msk, se);

% Naming output folder
if contains(input_workspace, 'ILR')
    output_folder = 'ILR_localfields';
elseif contains(input_workspace, 'pred')
    output_folder = 'pred_localfields';
else
    output_folder = 'HR_localfields'; % for High Resolution
end

% Loop over threshold values for TKD thresholding
for tkd_thresh = TKDs
    fprintf('TKD threshold : %.2f\n', tkd_thresh);
    
    for weight =  weights  % Loop over weights for central image
        if weight
            fprintf('Central Weighting : %d%%. \n', weight*100);
        end
        
        % Define output folder and create if it does not exist
        subfolder = sprintf('w%.1f_tkd%.2f', weight, tkd_thresh);
        ofolder =  fullfile(directory, 'results', output_folder, subfolder);
        
        if ~isfolder(ofolder)
            mkdir(ofolder);
        end
        
        % Logging
        logParameters(ofolder, input_workspace, tkd_thresh, appliedOri, weight);
        
        % Set datatype for NIfTI file
        info.Datatype = class(FL_all);
        
        % Save all data
        if ~exist('FL_all', 'var')
            FL_all = fl_all; % already in ppm
            R_TOT = R_tot;
        end
        
        % Get combinations of orientations
        d = struct('ori5' , 1:size(FL_all,4)); 
        d.ori3 = nchoosek(d.ori5,3);
        d.ori4 = nchoosek(d.ori5,4);
        
        if ~exist('C', 'var') % If not passed in input workspace
            C = 'CEFLR';
        end
        
        % Apply COSMOS algorithm with different parameters
        for i = appliedOri 
            disp(i);
            
            if i == "ori5"
                apply_cosmos(i, C, R_TOT, vsz, N, tkd_thresh, FL_all, msk, weight, ofolder, info)
            else 
                comb = d.(i);
                
                for t = 1:length(comb)
                    index = comb(t,:);
                    
                    if ismember(1, index) % Excecute only if it contains the central position
                        disp(C(index));
                        
                        fl_all = FL_all(:,:,:,index);
                        R_tot = R_TOT(:,:,index);

                        apply_cosmos(i, C(index), R_tot, vsz, N, tkd_thresh, fl_all, msk, weight, ofolder, info)
                    end
                end
            end
        end
    end
end


% Function to log parameters
function logParameters(ofolder, input_workspace, tkd_thresh, appliedOri, weight)

% Generates a log file with specified parameters.
%
%   Parameters:
%   - ofolder: Output folder for the log file.
%   - input_workspace: Path to the input workspace.
%   - tkd_thresh: Threshold value for TKD truncation.
%   - appliedOri: Applied orientations.
%   - weight: Weight assigned to the central image.
%
%   Example:
%   logParameters('path/to/output/folder', 'input_workspace.mat', 0.2, ["ori5", "ori4"], 0.5);

    logFileName = fullfile(ofolder, 'logfile.txt');
    fid = fopen(logFileName, 'w');

    fprintf(fid, '----------------------\n');
    fprintf(fid, 'Modified COSMOS-QSM : 5 head orientation data \n');
    fprintf(fid, '----------------------\n');
    fprintf(fid, 'Running file: %s\n', mfilename('fullpath'));
    fprintf(fid, 'Timestamp of code execution: %s\n', datetime('now'));
    fprintf(fid, 'Input workspace: %s\n', input_workspace);
   
    % Log the applied orientations
    fprintf(fid, '\nApplied Orientations:\n');
    for ori = appliedOri
        fprintf(fid, '%s\n', ori);
    end

     fprintf(fid, '\n----------------------\n');
    fprintf(fid, 'Parameters\n');
    fprintf(fid, '----------------------\n');

    if tkd_thresh
        fprintf(fid, 'TKD threshold : %.2f\n', tkd_thresh);
    else 
        fprintf(fid, 'No kernel thresholding was applied \n');
    end

    % Log the image weighting option
    if weight
        fprintf(fid, 'Image weighting was applied, giving higher weight to the central image compared to the complementary orientations. \n');
        fprintf(fid, 'Central image was weighted by %d%%. \n', weight*100);
    else
        fprintf(fid, 'No image weighting was applied, all orientations were treated equally. \n');
    end

    fclose(fid);
end

function apply_cosmos(i, orientations, R_tot, vsz, N, tkd_thresh, fl_all, msk, weight, ofolder, info)

% Applies the modified COSMOS algorithm with specified parameters.
%
%   Parameters:
%   - i: number of applied orientations ["ori5", "ori4", "ori3"]; 
%   - orientations: Letters of orientation used (e.g., CLR for central, left, right).
%   - R_tot: Total field map.
%   - vsz: Voxel size.
%   - N: Matrix size.
%   - tkd_thresh: Threshold value for TKD truncation.
%   - fl_all: Field maps for all orientations.
%   - msk: Mask for ROI.
%   - weight: Weight assigned to the central image.
%   - ofolder: Output folder.
%   - info: NIfTI file information.
%
%   Example:
%   apply_cosmos('ori5', 'CLR', R_tot, vsz, N, 0.2, fl_all, msk, 0.5, 'output_folder', niftiinfo('filename.nii'));
    
% Create kernel
    [kernel, perc] = createKernel(R_tot, vsz, N, tkd_thresh);

    % Apply cosmos function
    x = cosmos(fl_all, kernel, msk, weight);

    % Get datatype of the result
    info.Datatype = class(x);

    % Create filename
    if ~tkd_thresh && ~weight % If there is no weighting or thresholding
        filename = join(['cosmos_', i, '_', orientations, '.nii'], '');
    else
        filename = join(['mCosmos_', i, '_', orientations, '.nii'], '');
    end
    ofile = fullfile(ofolder, filename);

    % Write NIfTI file
    niftiwrite(x, ofile, info);
end

