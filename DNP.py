import torch
import utils
from tqdm import tqdm
from unet import Unet
import os
import numpy as np
import argparse


def optimize(model, criterion, input, target, samples_dir, LR, num_iter, sr, save_every, accumulator):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
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
        stft = np.abs(utils.torch_stft(out, nfft=accumulator.nfft, center=accumulator.center))
        accumulator.sum_difference(stft, j)
        if (j + 1) % save_every == 0:
            # clip & normalize mask
            accumulator.create_atten_map()
            accumulator.mmse_lsa()
            # save wiener denoised files
            utils.write_music_stft(accumulator.stft_noisy_filt * accumulator.atten_map,
                                   f'{samples_dir}/wiener_{j}.wav', sr, center=accumulator.center)
            utils.write_music_stft(accumulator.stft_noisy_filt * accumulator.lsa_mask,
                                   f'{samples_dir}/lsa_{j}.wav', sr, center=accumulator.center)
            # write net output
            out_write = out.clone().detach()
            out_write = out_write.detach().cpu().numpy()
            utils.write_norm_music(out_write, f'{samples_dir}/net_output_{j}.wav', sr)


def dnp(run_name, noisy_file, samples_dir, LR=0.001, num_iter=5000, save_every=50):

    # Initiate model
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

    # Initialize accumulator
    nfft = 512
    residual = 10 ** (-30 / 10)  # -18 db lower gain
    low_cut = 10
    high_cut = 90
    center = False
    bandpass = int(round(3 / 512 * nfft))
    accumulator = utils.Accumulator(target, low_cut, high_cut, nfft, center, residual, sr, bandpass)

    # Run the algorithm
    optimize(model, criterion, input, target, samples_dir, LR, num_iter, sr, save_every, accumulator)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='demo', help='name of the run')
    parser.add_argument('--LR', default=0.001, type=float)
    parser.add_argument('--num_iter', default=5000, type=int)
    parser.add_argument('--save_every', default=250, type=int)
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
