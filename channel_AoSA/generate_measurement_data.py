# -*- coding: utf-8 -*-
# @Author: wentao.yu
# @Date:   2023-02-24 23:12:59
# @Last Modified by:   wentao.yu
# @Last Modified time: 2023-02-25 15:41:24

from scipy.io import loadmat
from scipy.io import savemat
import numpy as np

def genMeasurements(x,A,W_RF_hermitian,SNR_all):

    [N,K] = np.shape(x)  # 1024, 80000
    [M,N] = np.shape(A)  # 512, 1024
    batch = len(SNR_all)   # 100
    batchsize = int(K/batch)   # K/batch = 80000/100 = 800
    y = np.zeros(dtype=np.complex128, shape=(M, K))   # 512, 80000
    sigma_squared = np.zeros(dtype=np.float128, shape=(1, K))   # 1, 80000
    for k in range(int(batch)):   # 100
        xk = x[..., k*batchsize:(k+1)*batchsize]   # 800 data points
        SNR_dB = SNR_all[k]
        print(SNR_dB)
        SNR = np.power(10, SNR_dB / 10.)
        sigma_squared_temp = 1 / SNR

        # complex Gaussian noise
        cNoise = np.sqrt(sigma_squared_temp) * ((np.random.normal(size=(N, batchsize)) + 1j * np.random.normal(size=(N, batchsize))) / np.sqrt(2))

        yk = np.matmul(A, xk) + np.matmul(W_RF_hermitian, cNoise)
        y[..., k*batchsize:(k+1)*batchsize] = yk
        sigma_squared[..., k*batchsize:(k+1)*batchsize] = sigma_squared_temp.repeat(batchsize).T

    return y, sigma_squared

if __name__ == '__main__':
    M = 512
    N = 1024
    type = 'AoSA'  # 'AoSA'

    D = loadmat('./dataset/THzUMHF_' + type + '_training_channel_300GHz_1024.mat')
    training_x = D['H']
    D = loadmat('./dataset/THzUMHF_' + type + '_validation_channel_300GHz_1024.mat')
    validation_x = D['H']
    D = loadmat('./dataset/THzUMHF_' + type + '_testing_channel_300GHz_1024.mat')
    testing_x = D['H']

    training_size = np.shape(training_x)
    validation_size = np.shape(validation_x)

    D = loadmat('./dataset/CSmatrix' + str(N) + '_' + str(M) +'_AoSA_angle'+'.mat')
    A = D['A']
    W_RF_hermitian = D['W_RF_hermitian']

    training_y = np.zeros(dtype=np.complex128, shape=(M,training_size[1]))
    training_sigma_squared = np.zeros(dtype=np.float128, shape=(training_size[1],))   # 80000, 1
    validation_y = np.zeros(dtype=np.complex128, shape=(M, validation_size[1]))
    validation_sigma_squared = np.zeros(dtype=np.float128, shape=(validation_size[1],))   # 80000, 1

    # -----------------------------------------------------------------------------
    # generate training set
    # -----------------------------------------------------------------------------
    n1 = 0
    n2 = 10
    num_SNR_levels = 100
    SNR_dB = np.random.uniform(n1, n2, size=num_SNR_levels)
    training_y, training_sigma_squared = genMeasurements(training_x, A, W_RF_hermitian, SNR_dB)
    D = dict(y=training_y, sigma_squared=training_sigma_squared)
    savemat('./dataset/THzUMHF_'+type+'_training_'+str(M)+'_measurements_'+str(n1)+'to'+str(n2)+'dB.mat', D)

    n1 = 10
    n2 = 20
    num_SNR_levels = 100
    SNR_dB = np.random.uniform(n1, n2, size=num_SNR_levels)
    training_y, training_sigma_squared = genMeasurements(training_x, A, W_RF_hermitian, SNR_dB)
    D = dict(y=training_y, sigma_squared=training_sigma_squared)
    savemat('./dataset/THzUMHF_'+type+'_training_'+str(M)+'_measurements_'+str(n1)+'to'+str(n2)+'dB.mat', D)

    # -----------------------------------------------------------------------------
    # generate validation set
    # -----------------------------------------------------------------------------
    n1 = 0
    n2 = 10
    num_SNR_levels = 100
    SNR_dB = np.random.uniform(n1, n2, size=num_SNR_levels)
    training_y, training_sigma_squared = genMeasurements(validation_x, A, W_RF_hermitian, SNR_dB)
    D = dict(y=training_y, sigma_squared=training_sigma_squared)
    savemat('./dataset/THzUMHF_'+type+'_validation_'+str(M)+'_measurements_'+str(n1)+'to'+str(n2)+'dB.mat', D)

    n1 = 10
    n2 = 20
    num_SNR_levels = 100
    SNR_dB = np.random.uniform(n1, n2, size=num_SNR_levels)
    training_y, training_sigma_squared = genMeasurements(validation_x, A, W_RF_hermitian, SNR_dB)
    D = dict(y=training_y, sigma_squared=training_sigma_squared)
    savemat('./dataset/THzUMHF_'+type+'_validation_'+str(M)+'_measurements_'+str(n1)+'to'+str(n2)+'dB.mat', D)

    # -----------------------------------------------------------------------------
    # generate testing set
    # -----------------------------------------------------------------------------
    n = [0]
    testing_y, testing_sigma_squared = genMeasurements(testing_x, A, W_RF_hermitian, n)
    D = dict(y=testing_y, sigma_squared=testing_sigma_squared)
    savemat('./dataset/THzUMHF_' + type + '_testing_' + str(M) + '_measurements_' + str(n[0])+'dB.mat', D)

    n = [5]
    testing_y, testing_sigma_squared = genMeasurements(testing_x, A, W_RF_hermitian, n)
    D = dict(y=testing_y, sigma_squared=testing_sigma_squared)
    savemat('./dataset/THzUMHF_' + type + '_testing_' + str(M) + '_measurements_' + str(n[0])+'dB.mat', D)

    n = [10]
    testing_y, testing_sigma_squared = genMeasurements(testing_x, A, W_RF_hermitian, n)
    D = dict(y=testing_y, sigma_squared=testing_sigma_squared)
    savemat('./dataset/THzUMHF_' + type + '_testing_' + str(M) + '_measurements_' + str(n[0])+'dB.mat', D)

    n = [15]
    testing_y, testing_sigma_squared = genMeasurements(testing_x, A, W_RF_hermitian, n)
    D = dict(y=testing_y, sigma_squared=testing_sigma_squared)
    savemat('./dataset/THzUMHF_' + type + '_testing_' + str(M) + '_measurements_' + str(n[0])+'dB.mat', D)

    n = [20]
    testing_y, testing_sigma_squared = genMeasurements(testing_x, A, W_RF_hermitian, n)
    D = dict(y=testing_y, sigma_squared=testing_sigma_squared)
    savemat('./dataset/THzUMHF_' + type + '_testing_' + str(M) + '_measurements_' + str(n[0])+'dB.mat', D)
