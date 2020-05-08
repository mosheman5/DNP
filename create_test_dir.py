import os
import shutil
import utils
from tqdm import tqdm
import numpy as np
import argparse

def create_test(main_dir, dest_dir, iter_num, template_po = 'power', type='.wav'):
    template = f'{template_po}_{iter_num}'
    template_name = template + type
    dest_dir = os.path.join(dest_dir, template)
    utils.makedirs(dest_dir)
    dirs = [dI for dI in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, dI))]
    for dir in tqdm(dirs):
        shutil.move(os.path.join(main_dir, dir, template_name), os.path.join(dest_dir, dir + type))


def create_test_samples(main_dir, dest_dir, iter_num, template_po = 'samples', type='.wav'):
    template = f'{template_po}_{iter_num}'
    template_name = template + type
    dest_dir = os.path.join(dest_dir, template)
    utils_dap.makedirs(dest_dir)
    dirs = [dI for dI in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, dI))]
    for dir in tqdm(dirs):
        shutil.move(os.path.join(main_dir, dir, template_name), os.path.join(dest_dir, dir + type))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='DAP_baseline', help='name of the run')
    opts = parser.parse_args()
    run_name = opts.run_name


    iter_list = np.linspace(250, 1000, 4, dtype=int)

    template_po_list = ['net_output', 'net_output_filt']

    for iter in iter_list:
        for template_po in tqdm(template_po_list):
            create_test(f'outputs/{run_name}',
                        f'outputs_edit/{run_name}', iter, template_po, type='.wav')
