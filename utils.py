# -*- coding: utf-8 -*-
# @Author: wentao.yu
# @Date:   2022-10-27 00:41:41
# @Last Modified by:   wentao.yu
# @Last Modified time: 2023-02-26 15:34:24

"""
Utility functions of the project. 

References: 
[1] W. Yu, Y. Shen, H. He, X. Yu, J. Zhang, and K. B. Letaief, “Hybrid far- and near-field channel estimation for THz ultra-massive MIMO via fixed point networks,” 
in Proc. IEEE Global Commun. Conf. (GLOBECOM), Rio de Janeiro, Brazil, Dec. 2022.
[2] W. Yu, Y. Shen, H. He, X. Yu, S. Song, J. Zhang, and K. B. Letaief, “An adaptive and robust deep learning framework for THz ultra-massive MIMO channel estimation,” 
arXiv preprint arXiv:2211.15939, 2022.
"""

import sys
sys.path.append(".") 
import torch
import torch.nn as nn
import time
import numpy as np
from scipy.io import loadmat
from prettytable import PrettyTable
from tqdm import tqdm
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_CS_matrix(num_measurements, num_antennas):
    global A
    # remember to change this to your own dataset name!
    A = loadmat('./dataset/CSmatrix' + str(num_antennas) + '_' + str(num_measurements) +'_AoSA_angle'+'.mat')['A']
    # change complex matrix A into the equivalent real matrix
    # A = |Re(A) -Im(A)|  (block_line1)
    #     |Im(A)  Re(A)|  (block_line2)
    block_line1 = np.hstack((np.real(A), -np.imag(A)))
    block_line2 = np.hstack((np.imag(A), np.real(A)))
    A = torch.from_numpy(np.vstack((block_line1, block_line2)))
    return A

def load_CEdataset(measurement_path, channel_path):
    # measurements (i.e., input)
    # change complex vector into the equivalent real form
    measurements = loadmat(measurement_path)['y']
    measurements = np.vstack((np.real(measurements), np.imag(measurements))).T
    measurements = torch.from_numpy(measurements).float()

    # channels (i.e., output)
    # change complex vector into the equivalent real form
    channels = loadmat(channel_path)['H']
    channels = np.vstack((np.real(channels), np.imag(channels))).T
    channels = torch.from_numpy(channels).float()

    # noise level (environmental noise variance, sigma_squared)
    # The noise levels are *not* used anywhere in the model. We write this only for the benefit of future extension. 
    noise_levels = loadmat(measurement_path)['sigma_squared'].T
    noise_levels = torch.from_numpy(noise_levels).float()
    return measurements, channels, noise_levels

class CEdataset():
    def __init__(self, dirname, array_type, num_measurements, t_SNR_range, v_SNR_range, train=True):
        super(CEdataset, self).__init__()
        if train == True:
            training_measurements = 'THzUMHF_' + array_type + '_training_' + str(num_measurements) + '_measurements_' + t_SNR_range + '.mat'
            training_channels = 'THzUMHF_' + array_type + '_training_channel_300GHz_1024.mat'   # remember to change this to your own dataset name!
            self.measurements, self.channels, self.noise_levels = load_CEdataset(f"{dirname}"+training_measurements, f"{dirname}"+training_channels)
        else:
            validation_measurements = 'THzUMHF_' + array_type + '_validation_' + str(num_measurements) + '_measurements_' + v_SNR_range + '.mat'
            validation_channels = 'THzUMHF_' + array_type + '_validation_channel_300GHz_1024.mat'   # remember to change this to your own dataset name!
            self.measurements, self.channels, self.noise_levels = load_CEdataset(f"{dirname}"+validation_measurements, f"{dirname}"+validation_channels)

    def __len__(self):
        return len(self.channels)

    def __getitem__(self, index):
        measurement = self.measurements[index]
        channel = self.channels[index]
        noise_level = self.noise_levels[index]
        return measurement, channel, noise_level

def dataloaders(dirname, array_type, num_measurements, t_SNR_range, v_SNR_range, train_batch_size, test_batch_size=None):
    train_dataset = CEdataset(dirname, array_type, num_measurements, t_SNR_range, v_SNR_range, train=True)
    test_dataset = CEdataset(dirname, array_type, num_measurements, t_SNR_range, v_SNR_range, train=False)

    if test_batch_size is None:
        test_batch_size = train_batch_size

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, drop_last=True)
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=test_batch_size, drop_last=True)
    return train_loader, test_loader

def get_stats(net, test_loader, criterion):
    """
    Reture the training status after each epoch. 
    """
    test_loss = 0.0

    with torch.no_grad():
        for y_test, h_test, sigma_squared_test in test_loader:
            h_test = h_test.to(net.device())
            y_test = y_test.to(net.device())
            sigma_squared_test = sigma_squared_test.to(net.device())
            batch_size = y_test.shape[0]

            h_predict = net(y_test)
            batch_loss = criterion(h_predict, h_test)
            test_loss += batch_size * batch_loss

    test_loss /= len(test_loader.dataset)
    test_NMSE = compute_NMSE(h_predict, h_test)
    
    return test_loss, test_NMSE

def model_params(net):
    table = PrettyTable(["Network Component", "# Parameters"])
    num_params = 0
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad:
            continue
        table.add_row([name, parameter.numel()])
        num_params += parameter.numel()
    table.add_row(['TOTAL', num_params])
    return table

def compute_NMSE(h_predict, h_label):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    h_predict = h_predict.to(device)
    h_label = h_label.to(device)
    batch_size = h_predict.shape[0]
    NMSE= 0.0
    for i in range(batch_size):
        numerator = torch.square(torch.norm((h_predict[i,:] - h_label[i,:]), p=2))
        denominator = torch.square(torch.norm(h_label[i,:], p=2))
        NMSE += numerator / denominator
    NMSE = NMSE / batch_size
    NMSE = 10 * torch.log10(NMSE)
    return NMSE

class NMSELoss(nn.Module):
    def __init__(self):
        super(NMSELoss, self).__init__()

    def forward(self, h_predict, h_label):
        batch_size = h_predict.shape[0]
        NMSE = 0.0
        for i in range(batch_size):
            numerator = torch.square(torch.norm((h_predict[i, :] - h_label[i, :]), p=2))
            denominator = torch.square(torch.norm(h_label[i, :], p=2))
            NMSE += numerator / denominator
        loss = NMSE / batch_size
        return loss

def train_FPN_OAMP(net, max_epochs, lr_scheduler, train_loader,
          validation_loader, optimizer, criterion, save_dir='./results'):
    """
    Train the proposed FPN-OAMP, save the checkpoints and training history
    """
    
    fmt = '[{:3d}/{:3d}]: train - ({:6.2f} dB, {:6.2e}), validation - ({:6.2f} dB, '
    fmt += '{:6.2e}) | depth = {:4.1f} | lr = {:5.1e} | time = {:4.1f} sec'

    depth_ave = 0.0
    best_validation_NMSE = 0.0

    total_time = 0.0
    time_hist = []
    validation_loss_hist = []
    validation_NMSE_hist = []
    train_loss_hist = []
    train_NMSE_hist = []

    print(net)
    print(model_params(net))

    # epoch level loop
    for epoch in range(max_epochs):
        time.sleep(0.5)  # slows progress bar so it won't print on multiple lines
        loss_ave = 0.0
        train_NMSE = 0.0
        epoch_start_time = time.time()
        tot = len(train_loader)

        with tqdm(total=tot, unit=" batch", leave=False, ascii=True) as tepoch:

            tepoch.set_description("[{:3d}/{:3d}]".format(epoch+1, max_epochs))

            # (mini-)batch level loop
            # enumerate(train_loader) will take out a batch of data for training in each epoch
            for _, (measurements, channels, _) in enumerate(train_loader):
                channels = channels.to(net.device())
                measurements = measurements.to(net.device())
                batch_size = measurements.shape[0]

                # specify the train mode
                net.train()
                
                # Apply network to get fixed point, and then backprop directly at the fixed point via fixed point theorem
                optimizer.zero_grad()
                channels_pred = net(measurements)

                depth_ave = 0.99 * depth_ave + 0.01 * net.depth
                output = None

                # loss function
                output = criterion(channels_pred, channels)
                loss_val = output.detach().cpu().numpy() * batch_size
                loss_ave += loss_val
                output.backward()
                optimizer.step()

                # Output training stats after each epoch
                train_NMSE = compute_NMSE(channels_pred, channels)
                tepoch.update(1)
                tepoch.set_postfix(train_loss="{:5.2e}".format(loss_val / batch_size),
                                   train_NMSE="{:f}".format(train_NMSE),
                                   depth="{:5.1f}".format(net.depth))

        loss_ave = loss_ave / len(train_loader.dataset)

        # use the evaluation mode for validation
        net.eval()

        validation_loss, validation_NMSE = get_stats(net,
                                         validation_loader,
                                         criterion)

        validation_loss_hist.append(validation_loss)
        validation_NMSE_hist.append(validation_NMSE)
        train_loss_hist.append(loss_ave)
        train_NMSE_hist.append(train_NMSE)

        epoch_end_time = time.time()
        time_epoch = epoch_end_time - epoch_start_time

        time_hist.append(time_epoch)
        total_time += time_epoch

        print(fmt.format(epoch+1, max_epochs, train_NMSE, loss_ave,
                         validation_NMSE, validation_loss, depth_ave,
                         optimizer.param_groups[0]['lr'],
                         time_epoch))
        
        # return to the train mode and continue training
        net.train()

        # save weights
        if validation_NMSE < best_validation_NMSE:
            best_validation_NMSE = validation_NMSE
            state = {
                'test_loss_hist': validation_loss_hist,
                'test_NMSE_hist': validation_NMSE_hist,
                'net_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler
            }
            file_name = save_dir + net.name() + '_weights.pth'
            torch.save(state, file_name)
            print('Model weights saved to ' + file_name)

        # save training history at the last epoch
        if epoch+1 == max_epochs:
            state = {
                'test_loss_hist': validation_loss_hist,
                'test_NMSE_hist': validation_NMSE_hist,
                'train_loss_hist': train_loss_hist,
                'train_NMSE_hist': train_NMSE_hist,
                'lr_scheduler': lr_scheduler,
                'time_hist': time_hist,
                'eps': net.eps,
            }
            file_name = save_dir + net.name() + '_history.pth'
            torch.save(state, file_name)
            print('Training history saved to ' + file_name)

        lr_scheduler.step()
        epoch_start_time = time.time()
    return net

def load_checkpoint(model, checkpoint_PATH, device):
    if checkpoint_PATH != None:
        model_CKPT = torch.load(checkpoint_PATH, map_location=torch.device(device))
        model.load_state_dict(model_CKPT['net_state_dict'])
        print('Checkpoint has been loaded!')
    return model

# test and the performance iteration-by-iteration and save relevant information
# This function only utilizes vanilla fixed point iteration. 
# You may also use any other acceleration algorithm for fixed point finding. 
def test_vs_iteration(net, y, h_label, depth_warning=False):
    ''' 
    test the NMSE performance vs number of iterations
    '''
    NMSE_hist = []
    res_norm_hist = []
    effective_error_hist = []
    output_LE_hist = []
    output_NLE_hist = []

    with torch.no_grad():
        depth = 0.0
        h = net.initialize_solution(y)
        h_prev = np.Inf * torch.ones(h.shape, device=net.device())
        early_stopping = False
        termination = False
        while not termination and depth < net.max_depth:
            h_prev = h.clone()
            h, output_LE_temp, output_NLE_temp = net.latent_space_forward(y, h)   # take care of the shape issue! 
            output_LE_hist.append(output_LE_temp)
            output_NLE_hist.append(output_NLE_temp)
            effective_error_hist.append(h_label.squeeze() - output_LE_temp.squeeze())
            NMSE = compute_NMSE(h, h_label)   # should be in dB scale
            NMSE_hist.append(NMSE.cpu().numpy())
            res_norm = torch.mean(torch.norm(h - h_prev, dim=1))
            if early_stopping:
                termination = (res_norm <= net.eps)
            res_norm_hist.append(res_norm)
            depth += 1.0
            
    if depth >= net.max_depth and depth_warning:
        print("\nWarning: Max Depth Reached - Break Forward Loop\n")

    return NMSE_hist, res_norm_hist, depth, effective_error_hist, output_LE_hist, output_NLE_hist

# test and the performance layer-by-layer and save relevant information
# This function only utilizes vanilla fixed point iteration. 
# You may also use any other acceleration algorithm for fixed point finding. 
def test_convergence(net, y, depth_warning=False):
    ''' 
    test the ResNorm vs number of layers (validating the linear convergence rate)
    '''
    res_norm_hist = []

    with torch.no_grad():
        # First, compute the fixed point h_star and save it
        depth = 0.0
        h = net.initialize_solution(y)

        while depth < net.max_depth:
            h, _, _ = net.latent_space_forward(y, h)
            depth += 1.0
        h_star = h.clone()   # h_star is the reference fixed point after a sufficient number of iterations
        
        # Then, compute the residual norm after each iteration
        depth = 0.0
        h = net.initialize_solution(y)

        while depth < net.max_depth:
            h, _, _ = net.latent_space_forward(y, h)
            res_norm = torch.mean(torch.square(torch.div(torch.norm(h - h_star, dim=1),torch.norm(h_star, dim=1))))   # starred
            res_norm_hist.append(res_norm.cpu().numpy())
            depth += 1.0
            
    if depth >= net.max_depth and depth_warning:
        print("\nWarning: Max Depth Reached - Break Forward Loop\n")

    return res_norm_hist, depth