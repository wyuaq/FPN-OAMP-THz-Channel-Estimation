%% Information about the project

% Author: wentao.yu
% Last modified time: 2023-02-26

% References: 
% [1] W. Yu, Y. Shen, H. He, X. Yu, J. Zhang, and K. B. Letaief, “Hybrid far- and near-field channel estimation for THz ultra-massive MIMO via fixed point networks,” 
% in Proc. IEEE Global Commun. Conf. (GLOBECOM), Rio de Janeiro, Brazil, Dec. 2022.
% [2] W. Yu, Y. Shen, H. He, X. Yu, S. Song, J. Zhang, and K. B. Letaief, “An adaptive and robust deep learning framework for THz ultra-massive MIMO channel estimation,” 
% arXiv preprint arXiv:2211.15939, 2022. 

%% generate_UPA_dictionary_matrix

function F = generate_UPA_dictionary_matrix(vertical_antennas,horizontal_antennas)
% This function generate the normalized DFT-based dictionary matrix for
% uniform planar array (UPA). It can also apply to uniform linear arrays
% (ULAs) by setting either the vertical or horizontal antenna to 1. 

% Input Arguments:
%   - The number of vertical antennas: vertical_antennas
%   - The number of horizontal_antennas: horizontal_antennas
%
% Output Arguments:
%   - The DFT-based dictionary matrix (for UPA): F
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DFT_matrix_vertical = (1/sqrt(vertical_antennas)) * dftmtx(vertical_antennas);
DFT_matrix_horizontal = (1/sqrt(horizontal_antennas)) * dftmtx(horizontal_antennas);
F = kron(DFT_matrix_vertical,DFT_matrix_horizontal);

end