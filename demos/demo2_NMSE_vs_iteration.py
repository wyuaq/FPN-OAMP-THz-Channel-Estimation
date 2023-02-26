# -*- coding: utf-8 -*-
# @Author: wentao.yu
# @Date:   2023-02-25 16:16:28
# @Last Modified by:   wentao.yu
# @Last Modified time: 2023-02-26 13:49:47

"""
demo2: Plot the NMSE performance as a function of the number of iterations. 

References: 
[1] W. Yu, Y. Shen, H. He, X. Yu, J. Zhang, and K. B. Letaief, “Hybrid far- and near-field channel estimation for THz ultra-massive MIMO via fixed point networks,” 
in Proc. IEEE Global Commun. Conf. (GLOBECOM), Rio de Janeiro, Brazil, Dec. 2022.
[2] W. Yu, Y. Shen, H. He, X. Yu, S. Song, J. Zhang, and K. B. Letaief, “An adaptive and robust deep learning framework for THz ultra-massive MIMO channel estimation,” 
arXiv preprint arXiv:2211.15939, 2022.
"""

import sys
sys.path.append(".") 
import torch
import numpy as np
from model import FPN_OAMP
from utils import load_CS_matrix, load_CEdataset, load_checkpoint, test_vs_iteration
import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('device = ', device)

# Specify the system and network parameters, initialize the network
num_measurements = 512
num_antennas = 1024
A = load_CS_matrix(num_measurements, num_antennas).float()
array_type = 'AoSA'
dirname = './dataset/'
testing_SNR = np.array([0,5,10,15,20])
testing_channels = 'THzUMHF_' + array_type + '_testing_channel_300GHz_1024.mat'
channel_path = dirname + testing_channels
lat_layers = 3
contraction_factor = 0.99
eps = 1e-2
max_depth = 15   # the max_depth during testing can be larger than that used in the training stage
structure = 'ResNet'
num_channels = 64

net = FPN_OAMP(A=A, lat_layers=lat_layers, contraction_factor=contraction_factor,
                 eps=eps, max_depth=max_depth, structure=structure, num_channels=num_channels)

# load the trained network and then test the performance
NMSE_hist_all = {}
res_norm_hist_all = {}
depth_all = {}
effective_error_hist_all = {}

for SNR in testing_SNR:
    if SNR < 10:
        checkpoint_PATH = './checkpoints/FPN_OAMP_ResNet_weights_0to10dB.pth'
    else:
        checkpoint_PATH = './checkpoints/FPN_OAMP_ResNet_weights_10to20dB.pth'

    testing_measurements = 'THzUMHF_' + array_type + '_testing_' + str(num_measurements) + '_measurements_' + str(SNR) + 'dB.mat'
    measurement_path = dirname + testing_measurements
    measurements, channels, _ = load_CEdataset(measurement_path, channel_path)
    measurements = measurements.to(device)
    channels = channels.to(device)

    # use part of the testing dataset for the demo, to avoid out-of-memory issue
    measurements = measurements[0:500,:]
    channels = channels[0:500,:]

    net = load_checkpoint(net, checkpoint_PATH, device).to(device)
    net.eval()

    NMSE_hist_all[SNR], res_norm_hist_all[SNR], depth_all[SNR], \
        effective_error_hist_all[SNR], _, _ = test_vs_iteration(net, measurements, channels, depth_warning=False)

# plot the NMSE performance vs the number of iterations
plt.switch_backend('agg')

plt.figure(1)
line_formats = ['ro-','go-','bo-','mo-','co-']

for i in range(len(testing_SNR)):
    NMSE_result = np.array(NMSE_hist_all[testing_SNR[i]][:])
    print(NMSE_result)
    plt.plot(range(1,int(depth_all[testing_SNR[i]]+1)), NMSE_hist_all[testing_SNR[i]], line_formats[i], label='SNR = '+str(testing_SNR[i])+'dB')

plt.xlim((1, 15))
plt.axis(xmin=1, xmax=15, ymin=-25, ymax=-5)
plt.xlabel('Number of iterations')
plt.ylabel('NMSE (dB)')
plt.legend()
plt.grid()
plt.show()

plt.savefig("./figures/demo2_NMSE_vs_iteration.jpg")