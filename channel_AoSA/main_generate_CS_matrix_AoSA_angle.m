%% Information about the project

% Author: wentao.yu
% Last modified time: 2023-02-26

% References: 
% [1] W. Yu, Y. Shen, H. He, X. Yu, J. Zhang, and K. B. Letaief, “Hybrid far- and near-field channel estimation for THz ultra-massive MIMO via fixed point networks,” 
% in Proc. IEEE Global Commun. Conf. (GLOBECOM), Rio de Janeiro, Brazil, Dec. 2022.
% [2] W. Yu, Y. Shen, H. He, X. Yu, S. Song, J. Zhang, and K. B. Letaief, “An adaptive and robust deep learning framework for THz ultra-massive MIMO channel estimation,” 
% arXiv preprint arXiv:2211.15939, 2022. 

%% Generate the measurement matrix for channel estimation

clc; clear;
N = 1024;   % number of antennas
N_RF = 4;   % number of RF chains
under_sampling_ratio = 0.5;   % i.e., (Q*N_RF)/N
M = round(N*under_sampling_ratio);   % length of pilot measurements
Q = M/N_RF;   % instances of pilot transmission

W_RF = [];
for i = 1:Q
    W_RF_q = [];
    for j = 1:N_RF
        w = (rand(N/N_RF,1)>0.5)*2-1;   % random one-bit pilot combiner
        W_RF_q = blkdiag(W_RF_q,w);
    end
    W_RF = [W_RF W_RF_q];
end
W_RF = sqrt((N) / (norm(W_RF,'fro')^2)) * W_RF;
W_RF_hermitian = W_RF';

F = generate_AoSA_dictionary_matrix(N,N_RF);   % far-field DFT dictionary

A = W_RF_hermitian * F;   % measurement matrix (denoted by M in the paper)

filename = ['../dataset/CSmatrix', num2str(N), '_', num2str(M), ...
    '_AoSA_angle.mat'];
save(filename, 'A','W_RF_hermitian');