# DNP
Audio Denoising with Deep Network Priors

This repository provides a PyTorch implementation of "Audio Denoising with Deep Network Priors"

The method trains on noisy audio signal and provides a the clean underlying signal. **Comparison with traditional unsupervised methods can be found [here](https://mosheman5.github.io/DNP/)**

The method is completely unsupervised and only trains on the specific audio clip that is being denoised.

The algorithm is based on the observation that modeling noise in the signal is harder that a clean signal. 
During the fitting we observe flactuations in different stages of the train. 
By calculating the amount of difference between outputs in the time-frequency domain we create a robust spectral mask used for denoising the noisy output. 

## Dependencies
A conda environment file is available in the repository.
* Python 3.6 +
* Pytorch 1.0
* Torchvision
* librosa
* tqdm
* scipy
* soundfile

## Usage

### 1. Cloning the repository & setting up conda environment
```
$ git clone https://github.com/mosheman5/DNP.git
$ cd DNP/
```
For creating and activating the conda environment:
```
$ conda env create -f environment.yml
$ source activate DNP
```
 
### 2. Testing

To test on the demo speech file:

```
$ python DNP.py --run_name demo --noisy_file demo.wav --samples_dir samples --save_every 50 --num_iter 5000 --LR 0.001
```

In order to test any other audio file insert the file path after the ```--noisy_file``` option.

The script saves ```.mat``` files of the mask which can be integrated to classical speech enhancement methods that can be found in this matlab [toolbox](https://www.crcpress.com/downloads/K14513/K14513_CD_Files.zip)

## Reference
If you found this code useful, please cite the following paper:
```TBD
```

## Acknowledgement
The implemantation of the network architecture is taken from [Wave-U-Net](https://github.com/f90/Wave-U-Net)
