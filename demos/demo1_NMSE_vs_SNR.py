# -*- coding: utf-8 -*-
# @Author: wentao.yu
# @Date:   2023-02-25 16:16:28
# @Last Modified by:   wentao.yu
# @Last Modified time: 2023-02-26 13:49:39

"""
demo1: Plot the NMSE performance as a function of SNR. 

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
import matplotlib.pyplot as plt
from model import FPN_OAMP
from utils import load_CS_matrix, load_CEdataset, compute_NMSE, load_checkpoint

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('device = ', device)

# specify the system and network parameters, initialize the network
num_measurements = 512
num_antennas = 1024
A = load_CS_matrix(num_measurements, num_antennas).float()
array_type = 'AoSA'
dirname = './dataset/'
testing_SNR = np.array([0,5,10,15,20])
testing_channels = 'THzUMHF_' + array_type + '_testing_channel_300GHz_1024.mat'
channel_path = dirname + testing_channels
testing_NMSE = np.zeros(len(testing_SNR))
lat_layers = 3
contraction_factor = 0.99
eps = 1e-2
max_depth = 15
structure = 'ResNet'
num_channels = 64

net = FPN_OAMP(A=A, lat_layers=lat_layers, contraction_factor=contraction_factor,
                 eps=eps, max_depth=max_depth, structure=structure, num_channels=num_channels)

# load the trained network and then test the performance
for i in range(len(testing_SNR)):
    if testing_SNR[i] < 10:
        checkpoint_PATH = './checkpoints/FPN_OAMP_ResNet_weights_0to10dB.pth'
    else:
        checkpoint_PATH = './checkpoints/FPN_OAMP_ResNet_weights_10to20dB.pth'
    net = load_checkpoint(net, checkpoint_PATH, device).to(device)
    net.eval()

    testing_measurements = 'THzUMHF_' + array_type + '_testing_' + str(num_measurements) + '_measurements_' + str(testing_SNR[i]) + 'dB.mat'
    measurement_path = dirname + testing_measurements
    measurements, channels, _ = load_CEdataset(measurement_path, channel_path)
    measurements = measurements.to(device)
    channels = channels.to(device)

    # use part of the testing dataset for the demo, to avoid out-of-memory issue
    measurements = measurements[0:500,:]
    channels = channels[0:500,:]

    # perform testing
    channels_pred = net(measurements)
    testing_NMSE[i] = compute_NMSE(channels_pred, channels)

# print the testing result in terms of NMSE
print(testing_NMSE)

# demo1: Plot the NMSE performance as a function of SNR
plt.switch_backend('agg')

plt.figure(1)
l1 = plt.plot(testing_SNR,testing_NMSE,'ro-',label='Proposed FPN-OAMP')
plt.xlabel('SNR (dB)')
plt.ylabel('NMSE (dB)')
plt.legend()
plt.grid()
plt.show()

plt.savefig("./figures/demo1_NMSE_vs_SNR.jpg")