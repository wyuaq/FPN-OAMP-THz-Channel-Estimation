%% Information about the project

% Author: wentao.yu
% Last modified time: 2023-02-26

% References: 
% [1] W. Yu, Y. Shen, H. He, X. Yu, J. Zhang, and K. B. Letaief, “Hybrid far- and near-field channel estimation for THz ultra-massive MIMO via fixed point networks,” 
% in Proc. IEEE Global Commun. Conf. (GLOBECOM), Rio de Janeiro, Brazil, Dec. 2022.
% [2] W. Yu, Y. Shen, H. He, X. Yu, S. Song, J. Zhang, and K. B. Letaief, “An adaptive and robust deep learning framework for THz ultra-massive MIMO channel estimation,” 
% arXiv preprint arXiv:2211.15939, 2022. 

%% array_response_spherical

function a = array_response_spherical(theta,phi,N,N_RF,d_a,d_sub,f_c,r_l)
% (The notations may be different from those used in the paper)
% Return the near-field spherical-wave array response vector a_near

% Input Arguments:
%   - Elevation AoA: theta
%   - azimuth A0A: phi
%   - number of antennas: N
%   - number of subarraies/RF chains: N_RF
%   - antenna spacing: d_a
%   - subarray spacing: d_sub
%   - carrier frequency: f_c
%   - distance between the l-th scatterer and the start of the array: r_l
%
% Output Arguments:
%   - near-field spherical-wave array response: a

% speed of light
c = 3e8;

a = zeros(sqrt(N),sqrt(N));

N1 = sqrt(N_RF);   % subarray index, n1
N2 = sqrt(N_RF);   % subarray index, n2
M1 = sqrt(N/N_RF);   % antenna index, m1
M2 = sqrt(N/N_RF);   % antenna index, m2
x = r_l * cos(phi)*sin(theta);
y = r_l * sin(phi)*sin(theta);
z = r_l * cos(theta);

for n1 = 1:N1
    for n2 = 1:N2
        for m1 = 1:M1
            for m2 = 1:M2
                length_subarray_x = (sqrt(N/N_RF)-1)*d_a;
                length_subarray_y = (sqrt(N/N_RF)-1)*d_a;
                position_x = (n1-1) * length_subarray_x + (n1-1) * d_sub + (m1-1) * d_a;
                position_y = (n2-1) * length_subarray_y + (n2-1) * d_sub + (m2-1) * d_a;
                under_sqrt = (x-position_x)^2 + (y-position_y)^2 + z^2;
                D = sqrt(under_sqrt);
                index_x = (n2-1) * sqrt(N/N_RF) + m2;
                index_y = (n1-1) * sqrt(N/N_RF) + m1;
                a(index_x,index_y) = (1/sqrt(N)) * exp(-1i*2*pi*f_c*D/c);
            end
        end
    end
end


