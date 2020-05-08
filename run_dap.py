from DNP import dnp
import os
from tqdm import tqdm
import random
import torch
import numpy as np


def run_dap(curr_num, parts_num, big_run_name):

    seed = 1234
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    noisy_dir = '../../data/speech_experiment/noisy_testset_wav_16kHz'
    outputs_dir = 'outputs'

    outputs_dir = os.path.join(outputs_dir, big_run_name)
    if not os.path.exists(outputs_dir):
        print("Creating directory: {}".format(outputs_dir))
        os.makedirs(outputs_dir)
    # dataset_list = os.listdir(noisy_dir)
    # dataset_list = [Id for Id in os.listdir(noisy_dir) if not os.path.isfile(os.path.join(outputs_dir, Id.split('.wav')[0], 'net_output_249.wav'))]
    dataset_list = [Id for Id in os.listdir(noisy_dir)]
    dataset_list = dataset_list[:len(dataset_list) - len(dataset_list) % parts_num]
    dataset_list = dataset_list[len(dataset_list)//parts_num * curr_num:len(dataset_list)//parts_num * (curr_num+1)]

    for it, vector in enumerate(tqdm(dataset_list)):
        noisy_file = os.path.join(noisy_dir, vector)
        run_name = vector.split('.wav')[0]
        dnp(run_name=run_name, noisy_file=noisy_file,
                        samples_dir=outputs_dir, LR=0.001, num_iter=1000, save_every=250)


if __name__ == '__main__':

    run_dap(curr_num=0, parts_num=4, big_run_name='DAP_baseline')







