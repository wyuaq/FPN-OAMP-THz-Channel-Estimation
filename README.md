# FPN-OAMP-THz-Channel-Estimation

## Introduction

This repository contains the codes of the fixed point network-based orthogonal approximate message passing (FPN-OAMP) algorithm proposed in our [paper](https://arxiv.org/pdf/2205.04944.pdf) "**Hybrid Far- and Near-Field Channel Estimation for THz Ultra-Massive MIMO via Fixed Point Networks**", which was accepted by the 2022 IEEE Global Communications Conference (Globecom'22). 

The extended [journal version](https://arxiv.org/pdf/2211.15939.pdf) "**An Adaptive and Robust Deep Learning Framework for THz Ultra-Massive MIMO Channel Estimation**" has been submitted to IEEE journal for possible publication. Codes will also be uploaded and merged to this repository upon acceptance. 

The notations used in the codes may be slightly different from those in the papers. I tried to clarify with detailed comments. Should you meet any confusion or need any further help, please feel free to contact me via email (you can find the address in the papers listed above). 

If you find the codes useful for your research, please kindly cite our papers (bibtex file is provided at the bottom). Enjoy reproducible research! 

## Prerequisites

- Matlab R2021b

- Python 3.8

- Pytorch 1.9.1

- CUDA 11.6 (other versions should also be fine)

## Getting started

### Step 1: Set the environment

Create a new conda environment and install dependencies. It is highly suggested to run the codes on GPU, though they work on CPU as well. 

```
conda create --name <your_environment_name> python=3.8
conda activate <your_environment_name>
pip install -r requirements.txt
```
### Step 2:  Prepare the dataset

Download the pre-generated datasets from [FPN-OAMP-dataset](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wyuaq_connect_ust_hk/EnjI6Aev9I5CpNCLdDTvaXQBef3i_gkapkmc8SFBRWsJYw?e=1fwI7o), and paste them into ./dataset/.

The datasets are quite large and may take some time to download. If you wish to generate your own datasets, you may do so by

> First, running ./channel_AoSA/main_generate_hybrid_field_channel.m
> 
> Second, running  ./channel_AoSA/main_generate_CS_matrix_AoSA_angle.m
> 
> Third, running ./channel_AoSA/generate_measurement_data.py

### Step 3: Download the checkpoints / Perform training

Download the pretrained checkpoints from [FPN-OAMP-checkpoints](https://hkustconnect-my.sharepoint.com/:f:/g/personal/wyuaq_connect_ust_hk/EnHPxyogdzRIvz4oQsLBU1EBjmuR9dFlzeTX2CjTw1Rbkw?e=hy61F9), and paste it into ./checkpoints/.

Two checkpoints are provided for download, one trained with SNR in [0,10] dB, and the other trained with SNR in [10,20] dB. We find that dividing the SNR into two ranges offers competitive performance and good generalization capability. 

If you wish to retrain the neural networks, you may do so by

> Running ./train.py

### Step 4: Perform testing

We provide three demos to test the performance in ./demos/. Demo1 evaluates the NMSE performance as a function of SNR. Demo2 evaluates the NMSE performance as a function of the number of iterations (convergence in terms of the objective function). Demo3 evaluates the residual norm as a function of the number of iterations (linear convergence rate to the unique fixed point). The relevant figures are available at ./figures/.

**Note:** the results reproduced by this repository are slightly better than those reported in our paper, due to the additional use of layer normalization. 

## Citation


```
@inproceedings{yu2022hybrid,
  author={Yu, Wentao and Shen, Yifei and He, Hengtao and Yu, Xianghao and Zhang, Jun and Letaief, Khaled B.},
  booktitle={GLOBECOM 2022 - 2022 IEEE Global Communications Conference}, 
  title={Hybrid Far- and Near-Field Channel Estimation for THz Ultra-Massive MIMO via Fixed Point Networks}, 
  year={2022},
  volume={},
  number={},
  pages={5384-5389},
  doi={10.1109/GLOBECOM48099.2022.10001564}}

@article{yu2022adaptive,
  title={An Adaptive and Robust Deep Learning Framework for THz Ultra-Massive MIMO Channel Estimation},
  author={Yu, Wentao and Shen, Yifei and He, Hengtao and Yu, Xianghao and Song, Shenghui and Zhang, Jun and Letaief, Khaled B},
  journal={arXiv preprint arXiv:2211.15939},
  year={2022}
}
```
