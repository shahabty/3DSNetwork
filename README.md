# 3DSNetwork
This is an implementation of paper "Occupancy Networks - Learning 3D Reconstruction in Function Space" that is different from the [official implementation](https://github.com/autonomousvision/occupancy_networks). It includes a CRF module and a different training, validation and testing code. 

### Installation

1. Install Anaconda3.

2. Run the following commands to create conda environment and install all dependencies:

```console
username@PC:~$ conda env create -f environment.yml
username@PC:~$ conda activate onet-crf
```
### Data Preparation
We follow the same data preparation described [here](https://github.com/autonomousvision/occupancy_networks)
### Training and Testing
In order to train and validate, cfg['mode'] must be 'train' in the main. Then:
```console
username@PC:~$ python main.py.
```
To test it, cfg['mode'] must be 'test'. Then:
```console
username@PC:~$ python main.py.
```

The output files and logs will be saved in cfg['out']['out_dir'].
