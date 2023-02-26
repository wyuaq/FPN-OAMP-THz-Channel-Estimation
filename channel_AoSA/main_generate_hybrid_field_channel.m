%% Information about the project

% Author: wentao.yu
% Last modified time: 2023-02-26

% References: 
% [1] W. Yu, Y. Shen, H. He, X. Yu, J. Zhang, and K. B. Letaief, “Hybrid far- and near-field channel estimation for THz ultra-massive MIMO via fixed point networks,” 
% in Proc. IEEE Global Commun. Conf. (GLOBECOM), Rio de Janeiro, Brazil, Dec. 2022.
% [2] W. Yu, Y. Shen, H. He, X. Yu, S. Song, J. Zhang, and K. B. Letaief, “An adaptive and robust deep learning framework for THz ultra-massive MIMO channel estimation,” 
% arXiv preprint arXiv:2211.15939, 2022. 

%% Generate the dataset for wideband THz channel estimation

clc; clear; close all;
N = 1024;   % number of antennas
N_RF = 4;   % number of RF chains
L = 5;   % number of paths
f_c = 300e9;   % carrier frequency
d = 30;   % LoS path length
r_min = 10;   % Scatter distance range: [r_min,r_max]
r_max = 25;   % Scatter distance range: [r_min,r_max]
num_training_samples = 80000;   % num. of THz UM-MIMO channels for training
num_validation_samples = 5000;  % num. of THz UM-MIMO channels for validation
num_testing_samples = 5000;  % num. of THz UM-MIMO channels for testing

scenario = [num2str(f_c/1e9), 'GHz_', num2str(N)];

%% Generate THz UM-MIMO channel dataset: 
% (Have a short break. This may take a moment.)

% generate the THz channel for training
H = zeros(N,num_training_samples);   % narrowband channel

for i = 1:num_training_samples
    % transform the channel to angular domain
    h = transform_by_subarray(generate_hybrid_field_channel(N,N_RF,L,f_c,d,r_min,r_max),N,N_RF);
    % append to generate the dataset
    H(:,i) = h;
end

filename = ['../dataset/THzUMHF_AoSA_training_channel_', scenario, '.mat'];
save(filename, 'H');

% generate the THz channel for validation 
H = zeros(N,num_validation_samples);   % narrowband channel

for i = 1:num_validation_samples
    % transform the channel to angular domain
    h = transform_by_subarray(generate_hybrid_field_channel(N,N_RF,L,f_c,d,r_min,r_max),N,N_RF);
    % append to generate the dataset
    H(:,i) = h;
end

filename = ['../dataset/THzUMHF_AoSA_validation_channel_', scenario, '.mat'];
save(filename, 'H');

% generate the THz channel for testing
H = zeros(N,num_testing_samples);   % narrowband channel

for i = 1:num_testing_samples
    % transform the channel to angular domain
    h = transform_by_subarray(generate_hybrid_field_channel(N,N_RF,L,f_c,d,r_min,r_max),N,N_RF);
    % append to generate the dataset
    H(:,i) = h;
end

filename = ['../dataset/THzUMHF_AoSA_testing_channel_', scenario, '.mat'];
save(filename, 'H');
