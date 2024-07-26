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
from model_eng import EC


def main(args):
    random.seed(args.rand_seed)  
    np.random.seed(args.rand_seed)  
    torch.manual_seed(args.rand_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.rand_seed)
        torch.cuda.manual_seed_all(args.rand_seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train = np.loadtxt(os.path.join(args.ckpt_path.split('/')[0], 'train_loss.txt')).reshape(-1, 6)
    val = np.loadtxt(os.path.join(args.ckpt_path.split('/')[0], 'val_loss.txt')).reshape(-1, 6)
    step = 1000 * np.arange(1, len(train) + 1)

    plt.title('Loss')
    plt.plot(step, train[:, 0])
    plt.plot(step, val[:, 0])
    plt.legend(['training', 'validation'])
    plt.xlabel('Steps')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.savefig(os.path.join(args.ckpt_path.split('/')[0], 'loss.png'))

    print('Source:', args.input_path)

    # get pitch
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
    f0_real = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1]).astype('float32')

    # get energy
    x_real = librosa.feature.rms(y=wav16k, frame_length=1024, hop_length=80)[0].astype('float32')
    x_real = np.clip(x_real, args.eng_min, args.eng_max)
    
    # get singer's id
    c_trg = None
    meta = pickle.load(open(args.train_pkl_path, 'rb'))['test']
    for row in meta:
        if row[0] == args.target_singer:
            c_trg = row[1]

    model_pc = PC(args)
    checkpoint = torch.load(args.pc_ckpt_path, map_location='cpu')
    model_pc.load_state_dict(checkpoint['state_dict'])
    model_pc = model_pc.to(args.device)

    model = EC(args)
    checkpoint = torch.load(args.ckpt_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(args.device)

    model_pc.eval()
    model.eval()
    with torch.no_grad():
        x_real = torch.from_numpy(x_real).unsqueeze(0).to(args.device)
        c_trg = torch.from_numpy(c_trg).unsqueeze(0).to(args.device)
        f0_real = torch.from_numpy(f0_real).unsqueeze(0).to(args.device)

        f0_input, f0_output = model_pc(f0_real, c_trg)
        f0_pred = model_pc.reverse_embedding(f0_output)
        x_input, x_output = model(x_real, c_trg, f0_pred)
        
        # Identity mapping loss.
        loss_id = F.binary_cross_entropy(x_output, x_input)

        # Energy reconstruction loss.
        x_pred = model.reverse_embedding(x_output)
        loss_er = F.mse_loss(
            (x_pred / args.eng_min).log10() / math.log10(args.eng_max / args.eng_min),
            (x_real / args.eng_min).log10() / math.log10(args.eng_max / args.eng_min)
        ).sqrt()

        # Fourier transform loss.
        win = torch.special.sinc(torch.arange(-1, 1.02, 0.02)).expand(x_real.shape[0], -1).to(args.device)
        wav2spec = torchaudio.transforms.Spectrogram(n_fft=80, win_length=80, hop_length=20, power=2).to(args.device)
        x_real_ft = torchaudio.functional.convolve(x_real.log10(), win, mode='valid') / win.sum(-1, keepdim=True)
        x_real_ft = x_real.log10() - F.pad(x_real_ft, (50, 50), 'replicate')
        for k in range(x_real_ft.shape[0]):
            x_real_ft[k][(x_real_ft[k] - x_real_ft[k].mean()).abs() > 2 * x_real_ft[k].std()] = x_real_ft[k].clone().mean()
        x_real_ft = wav2spec(x_real_ft)
        x_pred_ft = torchaudio.functional.convolve(x_pred.log10(), win, mode='valid') / win.sum(-1, keepdim=True)
        x_pred_ft = x_pred.log10() - F.pad(x_pred_ft, (50, 50), 'replicate')
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
        x_pred_vi = (x_pred_vi > args.ext_th_eng)
        x_pred_vi = torch.logical_and(x_pred_vi[:, :-1], x_pred_vi[:, 1:])
        x_pred_v = x_pred_ext.clone()
        x_pred_v = torch.diff(x_pred_v, dim=-1)
        x_pred_v[~x_pred_vi] = 0
        loss_sv = F.mse_loss(x_pred_v, torch.zeros_like(x_pred_v)).sqrt()

        loss = args.lambda_id * loss_id + args.lambda_er * loss_er + args.lambda_ft * loss_ft + args.lambda_ext * loss_ext + args.lambda_sv * loss_sv
        print(pd.DataFrame({
            'loss': [loss.cpu().numpy()],
            'loss_id': [loss_id.cpu().numpy()],
            'loss_er': [loss_er.cpu().numpy()],
            'loss_ft': [loss_ft.cpu().numpy()],
            'loss_ext': [loss_ext.cpu().numpy()],
            'loss_sv': [loss_sv.cpu().numpy()],
        }))

    plt.clf()
    plt.plot(x_real[0].cpu().numpy())
    plt.plot(x_pred[0].cpu().numpy())
    plt.xlabel('Frame')
    plt.ylabel('Energy')
    plt.legend(['input', 'output'], loc='upper right')
    plt.subplots_adjust()
    plt.savefig(os.path.join(args.ckpt_path.split('/')[0], 'result_eng.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Pitch conversion model configuration.
    parser.add_argument('--chs_grp', type=int, default=16)
    parser.add_argument('--dim_enc', type=int, default=128)
    parser.add_argument('--dim_neck', type=int, default=2)
    parser.add_argument('--freq', type=int, default=128)
    parser.add_argument('--dim_emb', type=int, default=128)
    parser.add_argument('--dim_dec', type=int, default=128)

    # Energy conversion model configuration.
    parser.add_argument('--lambda_id', type=float, default=1, help='weight for identity mapping loss')
    parser.add_argument('--lambda_er', type=float, default=10, help='weight for energy reconstruction loss')
    parser.add_argument('--lambda_ft', type=float, default=0.01, help='weight for fourier transform loss')
    parser.add_argument('--lambda_ext', type=float, default=0.01, help='weight for extent contour loss')
    parser.add_argument('--lambda_sv', type=float, default=0.01, help='weight for smooth vibrato loss')
    parser.add_argument('--chs_grp_eng', type=int, default=16)
    parser.add_argument('--dim_enc_eng', type=int, default=128)
    parser.add_argument('--dim_neck_eng', type=int, default=2)
    parser.add_argument('--freq_eng', type=int, default=128)
    parser.add_argument('--dim_emb_eng', type=int, default=128)
    parser.add_argument('--dim_dec_eng', type=int, default=128)
    parser.add_argument('--eng_bin', type=int, default=128)
    parser.add_argument('--eng_max', type=int, default=1)
    parser.add_argument('--eng_min', type=int, default=1e-4)
    parser.add_argument('--ext_th_eng', type=int, default=20)
    parser.add_argument('--f0_max', type=int, default=1100)
    parser.add_argument('--f0_min', type=int, default=50)

    # Training configuration.
    parser.add_argument('--input_path', type=str, default='data/m4singer/Tenor-7#送别/0001.wav') # path to source audio
    parser.add_argument('--target_singer', type=str, default='opencpop') # target singer name
    parser.add_argument('--train_pkl_path', type=str, default='train.pkl')
    parser.add_argument('--pc_ckpt_path', type=str, default='ckpt_f0/model_ckpt_steps_400000.ckpt')
    parser.add_argument('--ckpt_path', type=str, default='ckpt_eng/model_ckpt_steps_400000.ckpt')
    parser.add_argument('--device', type=torch.device, default='cuda')
    parser.add_argument('--rand_seed', type=int, default=2023)
    
    args = parser.parse_args()

    main(args)
