function X_cosmos = cosmos(phs_all, kernel, msk, weight)

% Applies the COSMOS with orientation weightinh algorithm to reconstruct
% susceptibility maps.
%
%   X_cosmos = cosmos(phs_all, kernel, msk, weight)
%
%   Parameters:
%   - phs_all: 4D array containing phase data for all orientations.
%   - kernel: 4D array containing the kernel information.
%   - msk: Binary mask indicating the region of interest.
%   - weight: Weight assigned to the central image.
%
%   Returns:
%   - X_cosmos: Reconstructed susceptibility map.
%
%   Example:
%   X_cosmos = cosmos(phase_data, kernel_data, binary_mask, 0.5);

    %Weighting
    W = ones(size(phs_all));
    if weight
        w = (1-weight)/(size(phs_all,4)-1);
        W(:,:,:, 1) = weight.*W(:,:,:,1);
        for i = 2: size(phs_all,4)
            W(:,:,:,i) = w.*W(:,:,:,i);
        end
    end
    
    Phase_all = zeros(size(phs_all)); 
    for t = 1:size(phs_all,4)
        % Fournier transform of phase data
        Phase_all(:,:,:,t) = fftn(phs_all(:,:,:,t));
    end
   
    kernel_sum = sum(W.*(abs(kernel).^2), 4);
    X_cosmos = real( ifftn( sum(W.*(kernel .* Phase_all), 4) ./ ( eps +kernel_sum) ) ) .* msk;
end