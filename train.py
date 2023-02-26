# -*- coding: utf-8 -*-
# @Author: wentao.yu
# @Date:   2023-02-25 14:53:09
# @Last Modified by:   wentao.yu
# @Last Modified time: 2023-02-26 13:51:10

"""
Train the proposed FPN-OAMP algorithm. 

References: 
[1] W. Yu, Y. Shen, H. He, X. Yu, J. Zhang, and K. B. Letaief, “Hybrid far- and near-field channel estimation for THz ultra-massive MIMO via fixed point networks,” 
in Proc. IEEE Global Commun. Conf. (GLOBECOM), Rio de Janeiro, Brazil, Dec. 2022.
[2] W. Yu, Y. Shen, H. He, X. Yu, S. Song, J. Zhang, and K. B. Letaief, “An adaptive and robust deep learning framework for THz ultra-massive MIMO channel estimation,” 
arXiv preprint arXiv:2211.15939, 2022.
"""

import torch
import torch.optim as optim
import numpy as np
from model import FPN_OAMP
from utils import dataloaders, train_FPN_OAMP
from utils import load_CS_matrix
from utils import NMSELoss

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print('device = ', device)

seed = np.random.randint(1,100)
print(seed)
torch.manual_seed(seed)
dataset_dir = './dataset/'
save_dir = './checkpoints/'
checkpt_path = './checkpoints/'

# neural network setup
num_measurements = 512
num_antennas = 1024
A = load_CS_matrix(num_measurements, num_antennas).float()   # measurement matrix (i.e., the M matrix in the papers)
lat_layers = 3
contraction_factor = 0.99
eps = 1e-2
max_depth = 15
structure = 'ResNet'
num_channels = 64   # num_channels of the intermediate feature maps

net = FPN_OAMP(A=A, lat_layers=lat_layers, contraction_factor=contraction_factor,
                 eps=eps, max_depth=max_depth, structure=structure, num_channels=num_channels).to(device)

# training setup
max_epochs = 150
learning_rate = 1e-3
weight_decay = 0
optimizer = optim.Adam(net.parameters(), lr=learning_rate,
                       weight_decay=weight_decay)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
loss = NMSELoss()

print('weight_decay = ', weight_decay, ', learning_rate = ', learning_rate,
      ', eps = ', eps, ', max_depth = ', max_depth, 'contraction_factor = ',
      contraction_factor, 'optimizer = Adam')

# load dataset
train_batch_size = 128
validation_batch_size = 2000
array_type = 'AoSA'
t_SNR_range = '0to10dB'   # change to '10to20dB' to train the high-SNR network
v_SNR_range = '0to10dB'   # change to '10to20dB' to train the high-SNR network

train_loader, validation_loader = dataloaders(dataset_dir, array_type, num_measurements, t_SNR_range, v_SNR_range, train_batch_size, validation_batch_size)

# start training
net = train_FPN_OAMP(net, max_epochs, lr_scheduler, train_loader,
                    validation_loader, optimizer, loss, save_dir)