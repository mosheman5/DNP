import torch
import utils
from tqdm import tqdm
from unet import Unet
import os
import numpy as np
import argparse
import scipy.io as sio


def optimize(parameters, model, criterion, input, target, samples_dir, LR, num_iter, sr, save_every):

    # define parameters
    nfft = 512
    residual = 10**(-18/10)   # -18 db lower gain
    low_cut = 10
    high_cut = 90
    stft_avg_prev = None
    center = False

    # stft of the noisy sample
    stft_full = utils.torch_stft(target, nfft=nfft, center=center)
    bandpass = int(round(3/512 * nfft))
    stft_full[:bandpass, :] = 0 * stft_full[:bandpass, :]  # reduce low frequencies
    stft_full[-bandpass // 3:, :] = 0 * stft_full[-bandpass // 3:, :]  # reduce high frequencies

    optimizer = torch.optim.Adam(parameters, lr=LR)
    input = input.view(1, 1, -1)

    for j in tqdm(range(num_iter)):
        # network step
        optimizer.zero_grad()
        out = model(input)
        out = out.view(-1)
        total_loss = criterion(out, target)
        total_loss.backward()
        optimizer.step()

        # accumulating the abs difference
        stft = np.abs(utils.torch_stft(out, nfft=nfft, center=center))
        if j < 50:
            stft_avg_prev, stft_avg_minus = stft, stft
            stft_minus_sum = np.zeros(stft.shape)
        else:
            stft_avg_minus = np.abs(stft - stft_avg_prev)/(stft+np.finfo(float).eps)
            stft_avg_minus[stft_avg_minus < np.percentile(stft_avg_minus, low_cut)] = np.percentile(stft_avg_minus, low_cut)
            stft_avg_minus[stft_avg_minus > np.percentile(stft_avg_minus, high_cut)] = np.percentile(stft_avg_minus, high_cut)
            stft_minus_sum += stft_avg_minus
            stft_avg_prev = stft


            if (j + 1) % save_every == 0 or j == 249:
                # clip & normalize mask
                max_mask = stft_minus_sum.max()
                min_mask = stft_minus_sum.min()
                atten_map = (max_mask - stft_minus_sum) / (max_mask - min_mask)
                atten_map[atten_map < residual] = residual

                utils.write_music_stft(stft_full * atten_map, f'{samples_dir}/wiener_{j}.wav', sr, f'wiener', center=center)
                utils.write_music_stft(stft_full * (1-atten_map), f'{samples_dir}/noise_wiener_{j}.wav', sr, center=center)

                # save the a-priori snr mask as .mat file, which can be integrated with classical speech-enhancement method
                atten_map_save = (atten_map.flatten('F'))
                sio.savemat(f'{samples_dir}/atten_map_{j}.mat', {'attenuation_vec': atten_map_save})

                out_write = out.clone().detach()
                out_write = out_write.detach().cpu().numpy()
                utils.write_norm_music(out_write, f'{samples_dir}/net_output_{j}.wav', sr)


def dnp(run_name, noisy_file, samples_dir, LR=0.001, num_iter=5000, save_every=50):

    # initiate model
    nlayers = 6
    model = Unet(nlayers=nlayers, nefilters=60).cuda()
    samples_dir = os.path.join(samples_dir, run_name)
    utils.makedirs(samples_dir)
    # load data
    target, sr = utils.load_wav_to_torch(noisy_file)
    target = target[:(len(target)//2**nlayers) * 2**nlayers]
    target = target/utils.MAX_WAV_VALUE
    input = torch.rand_like(target)
    input = (input - 0.5) * 2
    target, input = target.cuda(), input.cuda()
    criterion = torch.nn.MSELoss()

    optimize(model.parameters(), model, criterion, input, target, samples_dir, LR, num_iter, sr, save_every)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='demo', help='name of the run')
    parser.add_argument('--LR', default=0.001, type=float)
    parser.add_argument('--num_iter', default=5000, type=int)
    parser.add_argument('--save_every', default=5000, type=int)
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--noisy_file', type=str, default='demo.wav')
    parser.add_argument('--samples_dir', type=str, default='samples')

    opts = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    seed = opts.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    run_name = f'{opts.run_name}_{opts.LR}_{opts.num_iter}'
    dnp(run_name=run_name, noisy_file=opts.noisy_file, samples_dir=opts.samples_dir, LR=opts.LR, num_iter=opts.num_iter
        , save_every=opts.save_every)
