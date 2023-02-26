%% Information about the project

% Author: wentao.yu
% Last modified time: 2023-02-26

% References: 
% [1] W. Yu, Y. Shen, H. He, X. Yu, J. Zhang, and K. B. Letaief, “Hybrid far- and near-field channel estimation for THz ultra-massive MIMO via fixed point networks,” 
% in Proc. IEEE Global Commun. Conf. (GLOBECOM), Rio de Janeiro, Brazil, Dec. 2022.
% [2] W. Yu, Y. Shen, H. He, X. Yu, S. Song, J. Zhang, and K. B. Letaief, “An adaptive and robust deep learning framework for THz ultra-massive MIMO channel estimation,” 
% arXiv preprint arXiv:2211.15939, 2022. 

%% transform_by_subarray

function h = transform_by_subarray(H,N,N_RF)
% This function transforms the spatial AoSA channel to the angle domain, by
% using the DFT-based dictionary matrix in a subarray-by-subarray manner. 

length = sqrt(N/N_RF);
h = [];

for i = 1:sqrt(N_RF)
    for j =1:sqrt(N_RF)
        H_subarray = H((i-1)*length+1:i*length,(j-1)*length+1:j*length);

        vertical_antennas = sqrt(N/N_RF);
        horizontal_antennas = sqrt(N/N_RF);

        % generate the dictionary matrix *for each component UPA*
        F_UPA = generate_UPA_dictionary_matrix(vertical_antennas,horizontal_antennas);
        h_subarray = (F_UPA' * reshape(H_subarray, [N/N_RF 1])).';

        h = [h h_subarray];
    end
end

h = h.';


