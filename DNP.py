import torch
import torchaudio
import utils
from tqdm import tqdm
from model import UNet
import os
import numpy as np
import argparse


def optimize(model, criterion, input, target_stft, samples_dir, LR, num_iter, sr, save_every, nfft=1024):

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    input = input.unsqueeze(0)
    target_stft = target_stft.unsqueeze(0)

    for j in tqdm(range(num_iter)):
        # network step
        optimizer.zero_grad()
        out = model(input)
        total_loss = criterion(out, target_stft)
        total_loss.backward()
        optimizer.step()

        # write net output
        if j >0 and (j % save_every == 0):
            out_denorm = out.squeeze(0).detach() + 0
            out_write = torchaudio.functional.istft(out_denorm.permute(1, 2, 0), n_fft=nfft, hop_length=64)
            out_write = out_write.detach().cpu().numpy()
            utils.write_norm_music(out_write, f'{samples_dir}/net_output_{j+1}.wav', sr)
            bandpass = int(round(3 / 512 * nfft))
            out_denorm[:, :bandpass, :] = 0 * out_denorm[:, :bandpass, :]
            out_write = torchaudio.functional.istft(out_denorm.permute(1, 2, 0), n_fft=nfft, hop_length=64)
            out_write = out_write.detach().cpu().numpy()
            utils.write_norm_music(out_write, f'{samples_dir}/net_output_filt_{j}.wav', sr)


def dnp(run_name, noisy_file, samples_dir, LR=0.001, num_iter=5000, save_every=50):

    # Initiate model
    model = UNet().cuda()
    samples_dir = os.path.join(samples_dir, run_name)
    utils.makedirs(samples_dir)
    # load data
    target, sr = utils.load_wav_to_torch(noisy_file)
    target = target/utils.MAX_WAV_VALUE
    target_stft = torch.stft(target.cuda(), n_fft=1024, hop_length=64)
    target_stft = target_stft.permute(2, 0, 1)
    input = torch.randn_like(target_stft)
    input = input * 0.02
    input = input.cuda()
    criterion = torch.nn.MSELoss()

    # Run the algorithm
    optimize(model, criterion, input, target_stft, samples_dir, LR, num_iter, sr, save_every)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='demo', help='name of the run')
    parser.add_argument('--LR', default=0.001, type=float)
    parser.add_argument('--num_iter', default=1000, type=int)
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
