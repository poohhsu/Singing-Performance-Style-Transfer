import argparse
import os

import librosa
import numpy as np
from numpy.random import RandomState
import torch
import torchcrepe
from tqdm import tqdm


def main(args):
    if not os.path.exists(args.f0_output_dir):
        os.makedirs(args.f0_output_dir)
    if not os.path.exists(args.eng_output_dir):
        os.makedirs(args.eng_output_dir)

    for root, dirs, files in os.walk(args.input_dir):
        if 'TONAS/' not in root: # skip original TONAS dataset
            print(root)
            prng = RandomState(int(root.split('/')[-1]) if root.split('/')[-1].isnumeric() else 0)
            for file in tqdm(sorted(files)):
                if file.endswith('.wav'):
                    path = os.path.join(root, file)

                    if 1 in args.mode: # preprocess f0
                        wav16k, sr = librosa.load(path, sr=16000)
                        wav16k_torch = torch.FloatTensor(wav16k).unsqueeze(0).to(args.device)

                        f0, pd = torchcrepe.predict(wav16k_torch, 16000, 80, args.f0_min, args.f0_max, pad=True, model='full', batch_size=1024, device=args.device, return_periodicity=True)
                        pd = torchcrepe.filter.median(pd, 3)
                        pd = torchcrepe.threshold.Silence(-80.)(pd, wav16k_torch, 16000, 80)
                        f0 = torchcrepe.threshold.At(0.05)(f0, pd)
                        f0 = torchcrepe.filter.mean(f0, 3)
                        l = len(f0[0])

                        f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)
                        nzindex = torch.nonzero(f0[0]).squeeze()
                        f0 = torch.index_select(f0[0], dim=0, index=nzindex).cpu().numpy()
                        time_org = 0.005 * nzindex.cpu().numpy()
                        time_frame = np.arange(l) * 0.005
                        try:
                            f0 = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1])
                        except:
                            continue

                        if 'm4singer' in root:
                            save_path = os.path.join(args.f0_output_dir, f'm4singer_{root.split("/")[-1].split("#")[0]}')
                        elif 'opencpop' in root:
                            save_path = os.path.join(args.f0_output_dir, 'opencpop')
                        elif 'TONAS' in root:
                            save_path = os.path.join(args.f0_output_dir, 'TONAS')
                        # elif 'OpenSinger' in root: # for testing
                        #     save_path = os.path.join(args.f0_output_dir, f'OpenSinger_{root.split("/")[-1]}')

                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        save_path = os.path.join(save_path, file[:-4] if 'm4singer' not in root else f'{root.split("#")[-1]}_{file[:-4]}')
                        np.save(save_path, f0)

                    if 2 in args.mode: # preprocess energy
                        wav16k, sr = librosa.load(path, sr=16000)

                        eng = librosa.feature.rms(y=wav16k, frame_length=1024, hop_length=80)[0]

                        if 'm4singer' in root:
                            save_path = os.path.join(args.eng_output_dir, f'm4singer_{root.split("/")[-1].split("#")[0]}')
                        elif 'opencpop' in root:
                            save_path = os.path.join(args.eng_output_dir, 'opencpop')
                        elif 'TONAS' in root:
                            save_path = os.path.join(args.eng_output_dir, 'TONAS')
                        # elif 'OpenSinger' in root: # for testing
                        #     save_path = os.path.join(args.eng_output_dir, f'OpenSinger_{root.split("/")[-1]}')

                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        save_path = os.path.join(save_path, file[:-4] if 'm4singer' not in root else f'{root.split("#")[-1]}_{file[:-4]}')
                        np.save(save_path, eng)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/') # path to data
    parser.add_argument('--f0_output_dir', type=str, default='data_f0/')
    parser.add_argument('--eng_output_dir', type=str, default='data_eng/')
    parser.add_argument('--mode', type=int, nargs='+', default=[1, 2], help='1: f0, 2: eng')
    parser.add_argument('--device', type=torch.device, default='cuda')
    parser.add_argument('--f0_max', type=int, default=1100)
    parser.add_argument('--f0_min', type=int, default=50)

    args = parser.parse_args()

    main(args)