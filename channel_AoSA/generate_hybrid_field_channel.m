%% Information about the project

% Author: wentao.yu
% Last modified time: 2023-02-26

% References: 
% [1] W. Yu, Y. Shen, H. He, X. Yu, J. Zhang, and K. B. Letaief, “Hybrid far- and near-field channel estimation for THz ultra-massive MIMO via fixed point networks,” 
% in Proc. IEEE Global Commun. Conf. (GLOBECOM), Rio de Janeiro, Brazil, Dec. 2022.
% [2] W. Yu, Y. Shen, H. He, X. Yu, S. Song, J. Zhang, and K. B. Letaief, “An adaptive and robust deep learning framework for THz ultra-massive MIMO channel estimation,” 
% arXiv preprint arXiv:2211.15939, 2022. 

%% generate_hybrid_field_channel

function H = generate_hybrid_field_channel(N,N_RF,L,f_c,d,r_min,r_max)
% Input Arguments:
%   - antenna number: N
%   - Number of subarraies/RF chains: N_RF
%     (Number of antennas in each subarray: N/N_RF)
%   - number of paths: L
%   - carrier (central) frequency: f_c
%   - LoS path length: d
%   - Scatter distance range: [r_min,r_max]
%
% Output Arguments:
%   - Hybrid-field THz UM-MIMO channel: H (shape: sqrt(N)*sqrt(N))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% constants:
c = 3e8;   % speed of light
n_t = 2.24 - 0.025i;   % refractive index
sigma_rough = 0.088e-3;   % roughness factor
lambda_c = c/f_c;   % carrier wavelength
d_a = lambda_c/5;   % antenna spacing (half carrier wavelength)
d_sub = 56 * lambda_c;   % subarray spacing (56 carrier wavelength, widely spaced)
k_abs = 0.0033;   % molecular absorption coefficient
array_apperture = sqrt(2) * ((sqrt(N/N_RF)-1) * d_a * sqrt(N_RF) + (sqrt(N_RF)-1) * d_sub);   % array apperture
Rayleigh_distance = 2 * (array_apperture)^2/lambda_c;   % Rayleigh distance

% random variables:
delay_spread = 0.1*d/c;   % the delay spread used in the simulation
tau = unifrnd(d/c,d/c+delay_spread,[L 1]);   % delay of NLoS paths
tau(1) = d/c;   % delay of the LoS path
r_l = unifrnd(r_min,r_max,[L 1]);   % distance from the l-th scatterer to the center of the array
parphi_in = (pi/2)*rand(L,1);   % incident angle (only valid for NLoS paths, i.e., when l > 1)
parphi_ref = asin((1/n_t)*sin(parphi_in));   % refraction angle (only valid for NLoS paths, i.e., when l > 1)
Gamma = zeros(L,1);   % refraction loss, to be calculated later (only valid for NLoS paths, i.e., when l > 1)
% generate AoA
theta = pi*rand(L,1)-pi/2;   % Elevation AoA
phi = 2*pi*rand(L,1)-pi;   % Azimuth AoA

% empty channel (size: sqrt(N)*sqrt(N))
H = zeros(sqrt(N),sqrt(N));
alpha = zeros(L,1);   % path loss (=spread loss + absorption loss)

% loop over different paths
for l = 1:L
    if l == 1   % one LoS path
        L_spread = c / (4*pi*f_c*d);   % LoS spread loss
        L_abs = exp(-0.5*k_abs*d);   % LoS absorption loss
        alpha(l) = L_spread * L_abs;   % LoS path loss
        if d > Rayleigh_distance   % planar wave
            H = H + alpha(l) * array_response_planar(theta(l),phi(l),N,N_RF,d_a,d_sub,f_c) * exp(-1i*2*pi*f_c*tau(l));
        else   % spherical wave
            H = H + alpha(l) * array_response_spherical(theta(l),phi(l),N,N_RF,d_a,d_sub,f_c,d) * exp(-1i*2*pi*f_c*tau(l));
        end

    else   % (L-1) NLoS paths
        gamma = (cos(parphi_in(l)) - n_t*cos(parphi_ref(l))) / (cos(parphi_in(l)) + n_t*cos(parphi_ref(l)));
        exp_factor = -8*(pi^2)*(f_c^2)*(sigma_rough^2)*(cos(parphi_in(l))^2) / (c^2);
        rho = exp(exp_factor);
        Gamma(l) = gamma * rho;   % refractive coefficient of the l-th pass
        alpha(l) = abs(Gamma(l)) * alpha(1);   % NLoS path loss
        if r_l(l) > Rayleigh_distance   % planar wave
            H = H + alpha(l) * array_response_planar(theta(l),phi(l),N,N_RF,d_a,d_sub,f_c) * exp(-1i*2*pi*f_c*tau(l));
        else   % spherical wave
            H = H + alpha(l) * array_response_spherical(theta(l),phi(l),N,N_RF,d_a,d_sub,f_c,r_l(l)) * exp(-1i*2*pi*f_c*tau(l));
        end
    end

end

% Normalize the channel, but remain the variations over different antennas.
% (Note that normalizing the channel before or after the DFT are the same,
% due to the norm preserving property of unitary matrices.)
H = sqrt((N) / (norm(H,'fro')^2)) * H;



