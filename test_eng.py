import argparse
import os
import pickle
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchcrepe

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

    # predict
    model_pc.eval()
    model.eval()
    with torch.no_grad():
        f0_real = torch.from_numpy(f0_real).unsqueeze(0).to(args.device)
        x_real = torch.from_numpy(x_real).unsqueeze(0).to(args.device)
        c_trg = torch.from_numpy(c_trg).unsqueeze(0).to(args.device)

        f0_input, f0_output = model_pc(f0_real, c_trg)
        f0_pred = model_pc.reverse_embedding(f0_output)
        x_input, x_output = model(x_real, c_trg, f0_pred)
        x_pred = model.reverse_embedding(x_output)

    plt.plot(x_real[0].cpu().numpy())
    plt.plot(x_pred[0].cpu().numpy())
    plt.xlabel('Frame')
    plt.ylabel('Energy')
    plt.legend(['input', 'output'], loc='upper right')
    plt.subplots_adjust()
    plt.savefig('result_eng.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Pitch conversion model configuration.
    parser.add_argument('--chs_grp', type=int, default=16)
    parser.add_argument('--dim_enc', type=int, default=128)
    parser.add_argument('--dim_neck', type=int, default=2)
    parser.add_argument('--freq', type=int, default=128)
    parser.add_argument('--dim_emb', type=int, default=128)
    parser.add_argument('--dim_dec', type=int, default=128)
    parser.add_argument('--f0_max', type=int, default=1100)
    parser.add_argument('--f0_min', type=int, default=50)

    # Energy conversion model configuration.
    parser.add_argument('--chs_grp_eng', type=int, default=16)
    parser.add_argument('--dim_enc_eng', type=int, default=128)
    parser.add_argument('--dim_neck_eng', type=int, default=2)
    parser.add_argument('--freq_eng', type=int, default=128)
    parser.add_argument('--dim_emb_eng', type=int, default=128)
    parser.add_argument('--dim_dec_eng', type=int, default=128)
    parser.add_argument('--eng_bin', type=int, default=128)
    parser.add_argument('--eng_max', type=int, default=1)
    parser.add_argument('--eng_min', type=int, default=1e-4)

    # Training configuration.
    parser.add_argument('--input_path', type=str, default='data/test_audio.wav') # path to source audio
    parser.add_argument('--target_singer', type=str, default='opencpop') # target singer name
    parser.add_argument('--train_pkl_path', type=str, default='train.pkl')
    parser.add_argument('--pc_ckpt_path', type=str, default='ckpt_f0/model_ckpt_steps_400000.ckpt')
    parser.add_argument('--ckpt_path', type=str, default='ckpt_eng/model_ckpt_steps_400000.ckpt')
    parser.add_argument('--device', type=torch.device, default='cuda')
    parser.add_argument('--rand_seed', type=int, default=2023)
    
    args = parser.parse_args()

    main(args)
