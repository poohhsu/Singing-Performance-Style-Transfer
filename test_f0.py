import argparse
import math
import os
import pickle
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchcrepe
import torch.nn.functional as F

from model_f0 import PC


def main(args):
    random.seed(args.rand_seed)  
    np.random.seed(args.rand_seed)  
    torch.manual_seed(args.rand_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.rand_seed)
        torch.cuda.manual_seed_all(args.rand_seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train = np.loadtxt(os.path.join(args.ckpt_path.split('/')[0], 'train_loss.txt')).reshape(-1, 7)
    val = np.loadtxt(os.path.join(args.ckpt_path.split('/')[0], 'val_loss.txt')).reshape(-1, 7)
    step = 1000 * np.arange(1, len(train) + 1)

    plt.title('Loss')
    plt.plot(step, train[:, 0])
    plt.plot(step, val[:, 0])
    plt.legend(['training', 'validation'])
    plt.xlabel('Steps')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.savefig(os.path.join(args.ckpt_path.split('/')[0], 'loss.png'))

    print('Source:', args.input_path)
    wav16k, sr = librosa.load(args.input_path, sr=16000)
    wav16k_torch = torch.FloatTensor(wav16k).unsqueeze(0).to(args.device)

    f0, prd = torchcrepe.predict(wav16k_torch, 16000, 80, args.f0_min, args.f0_max, pad=True, model='full', batch_size=1024, device=args.device, return_periodicity=True)
    prd = torchcrepe.filter.median(prd, 3)
    prd = torchcrepe.threshold.Silence(-80.)(prd, wav16k_torch, 16000, 80)
    f0 = torchcrepe.threshold.At(0.05)(f0, prd)
    f0 = torchcrepe.filter.mean(f0, 3)
    l = len(f0[0])

    f0 = torch.where(torch.isnan(f0), torch.full_like(f0, 0), f0)
    nzindex = torch.nonzero(f0[0]).squeeze()
    f0 = torch.index_select(f0[0], dim=0, index=nzindex).cpu().numpy()
    time_org = 0.005 * nzindex.cpu().numpy()
    time_frame = np.arange(l) * 0.005
    x_real = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1]).astype('float32')
    
    c_trg = None
    meta = pickle.load(open(args.train_pkl_path, 'rb'))['test']
    for row in meta:
        if row[0] == args.target_singer:
            c_trg = row[1]

    model = PC(args)
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(args.device)

    model.eval()
    with torch.no_grad():
        x_real = torch.from_numpy(x_real).unsqueeze(0).to(args.device)
        c_trg = torch.from_numpy(c_trg).unsqueeze(0).to(args.device)
        x_input, x_output = model(x_real, c_trg)
            
        # Identity mapping loss.
        loss_id = F.binary_cross_entropy(x_output, x_input)

        # Pitch reconstruction loss.
        x_pred = model.reverse_embedding(x_output)
        loss_pr = F.mse_loss(
            (x_pred / args.f0_min).log2() / math.log2(args.f0_max / args.f0_min),
            (x_real / args.f0_min).log2() / math.log2(args.f0_max / args.f0_min)
        ).sqrt()

        # Fourier transform loss.
        win = torch.special.sinc(torch.arange(-1, 1.02, 0.02)).expand(1, -1).to(args.device)
        wav2spec = torchaudio.transforms.Spectrogram(n_fft=80, win_length=80, hop_length=20, power=2).to(args.device)
        x_real_ft = torchaudio.functional.convolve(x_real.log2(), win, mode='valid') / win.sum(-1, keepdim=True)
        x_real_ft = x_real.log2() - F.pad(x_real_ft, (50, 50), 'replicate')
        for k in range(x_real_ft.shape[0]):
            x_real_ft[k][(x_real_ft[k] - x_real_ft[k].mean()).abs() > 2 * x_real_ft[k].std()] = x_real_ft[k].clone().mean()
        x_real_ft = wav2spec(x_real_ft)
        x_pred_ft = torchaudio.functional.convolve(x_pred.log2(), win, mode='valid') / win.sum(-1, keepdim=True)
        x_pred_ft = x_pred.log2() - F.pad(x_pred_ft, (50, 50), 'replicate')
        for k in range(x_pred_ft.shape[0]):
            x_pred_ft[k][(x_pred_ft[k] - x_pred_ft[k].mean()).abs() > 2 * x_pred_ft[k].std()] = x_pred_ft[k].clone().mean()
        x_pred_ft = wav2spec(x_pred_ft)
        loss_ft = F.mse_loss(x_pred_ft, x_real_ft).sqrt()

        # Extent contour loss.
        f_idx = 40 / (16000 / 80 / 2)
        x_real_ext = x_real_ft[:, math.floor(5 * f_idx): math.ceil(8 * f_idx)].max(1)[0]
        x_pred_ext = x_pred_ft[:, math.floor(5 * f_idx): math.ceil(8 * f_idx)].max(1)[0]
        loss_ext = F.mse_loss(x_pred_ext, x_real_ext).sqrt()
                        
        # Smooth vibrato loss.
        x_pred_vi = x_pred_ext.clone()
        x_pred_vi = (x_pred_vi > args.ext_th)
        x_pred_vi = torch.logical_and(x_pred_vi[:, :-1], x_pred_vi[:, 1:])
        x_pred_v = x_pred_ext.clone()
        x_pred_v = torch.diff(x_pred_v, dim=-1)
        x_pred_v[~x_pred_vi] = 0
        loss_sv = F.mse_loss(x_pred_v, torch.zeros_like(x_pred_v)).sqrt()

        loss = args.lambda_id * loss_id + args.lambda_pr * loss_pr + args.lambda_ft * loss_ft + args.lambda_ext * loss_ext + args.lambda_sv * loss_sv
        print(pd.DataFrame({
            'loss': [loss.cpu().numpy()],
            'loss_id': [loss_id.cpu().numpy()],
            'loss_pr': [loss_pr.cpu().numpy()],
            'loss_ft': [loss_ft.cpu().numpy()],
            'loss_ext': [loss_ext.cpu().numpy()],
            'loss_sv': [loss_sv.cpu().numpy()],
        }))

    plt.clf()
    plt.plot(x_real[0].cpu().numpy())
    plt.plot(x_pred[0].detach().cpu().numpy())
    plt.xlabel('Frame')
    plt.ylabel('Frequency (Hz)')
    plt.legend(['input', 'output'])
    plt.savefig(os.path.join(args.ckpt_path.split('/')[0], 'result_f0.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--lambda_id', type=float, default=1, help='weight for identity mapping loss')
    parser.add_argument('--lambda_pr', type=float, default=10, help='weight for pitch reconstruction loss')
    parser.add_argument('--lambda_ft', type=float, default=0.1, help='weight for fourier transform loss')
    parser.add_argument('--lambda_ext', type=float, default=0.1, help='weight for extent contour loss')
    parser.add_argument('--lambda_sv', type=float, default=0.1, help='weight for smooth vibrato loss')
    parser.add_argument('--chs_grp', type=int, default=16)
    parser.add_argument('--dim_enc', type=int, default=128)
    parser.add_argument('--dim_neck', type=int, default=2)
    parser.add_argument('--freq', type=int, default=128)
    parser.add_argument('--dim_emb', type=int, default=128)
    parser.add_argument('--dim_dec', type=int, default=128)
    parser.add_argument('--f0_max', type=int, default=1100)
    parser.add_argument('--f0_min', type=int, default=50)
    parser.add_argument('--ext_th', type=int, default=0.75)

    # Training configuration.
    parser.add_argument('--input_path', type=str, default='data/m4singer/Tenor-7#送别/0001.wav') # path to source audio
    parser.add_argument('--target_singer', type=str, default='opencpop') # target singer name
    parser.add_argument('--train_pkl_path', type=str, default='train.pkl')
    parser.add_argument('--ckpt_path', type=str, default='ckpt_f0/model_ckpt_steps_400000.ckpt')
    parser.add_argument('--device', type=torch.device, default='cuda')
    parser.add_argument('--rand_seed', type=int, default=2023)
    
    args = parser.parse_args()

    main(args)
