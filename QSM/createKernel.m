function  [kernel, perc]= createKernel(R_tot, spatial_res, N, thresh_tkd)

% Generates a susceptibility thresholed kernel for COSMOS reconstruction.
%
%   Parameters:
%   - R_tot: 3x3xN matrix containing the orientations of the head.
%   - spatial_res: 1x3 vector containing the spatial resolution in each dimension.
%   - N: 1x3 vector containing the matrix size in each dimension.
%   - thresh_tkd: Threshold value for TKD truncation.
%
%   Returns:
%   - kernel: 4D array representing the susceptibility kernel for each orientation.
%   - perc: Percentage of the kernel that was thresholded.
%
%   Example:
%   [kernel, perc] = createKernel(orientation_matrix, [1, 1, 1], [128, 128, 128], 0.2);
    
    n_orientations = size(R_tot,3);
    [ky, kx, kz] = meshgrid(-N(2)/2:N(2)/2-1, -N(1)/2:N(1)/2-1, -N(3)/2:N(3)/2-1);

    kx = (kx / max(abs(kx(:)))) / spatial_res(1);
    ky = (ky / max(abs(ky(:)))) / spatial_res(2);
    kz = (kz / max(abs(kz(:)))) / spatial_res(3);
    k2 = kx.^2 + ky.^2 + kz.^2;
    
    kernel = zeros([N, n_orientations]); % matrice (dimx, dimy, dimz, nb_orientation)
    
    msk = ones(N);
    for t = 1:n_orientations
        kernel(:,:,:,t) = fftshift( 1/3 - (kx * R_tot(3,1,t) + ky * R_tot(3,2,t) + kz * R_tot(3,3,t)).^2 ./ (k2 + eps) ); 
        
        % Threshold only when values for all orientations < 1
        if (thresh_tkd~= 0)
            % get thresholding mask 4D
            msk = msk.*(abs(kernel(:,:,:,t))<thresh_tkd);
        end
    end
   
    if (thresh_tkd~= 0)
        % Generate 3D mask from 4D mask (with bool and multiplication)
        for t=2:n_orientations
            kernel3D = kernel(:,:,:,t);
            Kernel = kernel3D;
            
            Kernel(logical(msk)) = thresh_tkd*sign(kernel3D(logical(msk)));
            Kernel(kernel3D==0) = 0;
            kernel(:,:,:,t) = Kernel;
        end
        perc = nnz(msk)/numel(msk);
        fprintf('Percentage of kernel thresholded: %.2f%%.\n',perc*100);
    else 
        perc = 0;
    end
end