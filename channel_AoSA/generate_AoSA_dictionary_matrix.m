%% Information about the project

% Author: wentao.yu
% Last modified time: 2023-02-26

% References: 
% [1] W. Yu, Y. Shen, H. He, X. Yu, J. Zhang, and K. B. Letaief, “Hybrid far- and near-field channel estimation for THz ultra-massive MIMO via fixed point networks,” 
% in Proc. IEEE Global Commun. Conf. (GLOBECOM), Rio de Janeiro, Brazil, Dec. 2022.
% [2] W. Yu, Y. Shen, H. He, X. Yu, S. Song, J. Zhang, and K. B. Letaief, “An adaptive and robust deep learning framework for THz ultra-massive MIMO channel estimation,” 
% arXiv preprint arXiv:2211.15939, 2022. 

%% generate_AoSA_dictionary_matrix

function F = generate_AoSA_dictionary_matrix(N,N_RF)
% This function generate the normalized DFT-based far-field dictionary 
% matrix for array-of-subarray (AoSA). 

% Input Arguments:
%   - The number of antennas: N
%   - The number of RF chains: N_RF
%
% Output Arguments:
%   - The DFT-based dictionary matrix for AoSA: F
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The size of each component UPA in the AoSA is:
% "vertical_antennas * horizontal_antennas"

% For each component UPA, the DFT-based dictionary matrix is the Kronecker
% product of two smaller DFT matrices of the vertical and horizontal axises. 

vertical_antennas = sqrt(N/N_RF);
horizontal_antennas = sqrt(N/N_RF);
F_UPA = generate_UPA_dictionary_matrix(vertical_antennas,horizontal_antennas);

% align the UPA-dictionary matrices along the diagonal, to form the
% dictionary matrix for the AoSA. 
F = [];
for i = 1:N_RF
    F = blkdiag(F,F_UPA);
end

end