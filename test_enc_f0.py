import argparse
import os
import pickle
import random

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchcrepe

from model_enc_f0 import *
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
    x_real = np.interp(time_frame, time_org, f0, left=f0[0], right=f0[-1]).astype('float32')

    # get singer's id
    c_trg, i_trg = None, None
    meta = sorted(pickle.load(open(args.train_pkl_path, 'rb'))['test'])
    for i, row in enumerate(meta):
        if row[0] == args.target_singer:
            c_trg = row[1]
            i_trg = i

    model_pc = PC(args)
    checkpoint = torch.load(args.pc_ckpt_path, map_location='cpu')
    model_pc.load_state_dict(checkpoint['state_dict'])
    model_pc = model_pc.to(args.device)

    model = ResNetSE1D(args).to(args.device)
    checkpoint = torch.load(os.path.join(args.ckpt_dir, f'{args.model_name}.ckpt'))
    model.load_state_dict(checkpoint['state_dict'])

    # load average embeddings
    with open(os.path.join(args.ckpt_dir, f'emb_{args.model_name.split("_")[0]}.pkl'), 'rb') as f:
        idx2emb = pickle.load(f)

    model_pc.eval()
    model.eval()
    with torch.no_grad():
        x_real = torch.from_numpy(x_real).unsqueeze(0).to(args.device)
        c_trg = torch.from_numpy(c_trg).unsqueeze(0).to(args.device)

        x_input, x_output = model_pc(x_real, c_trg)
        x_pred_conv = model_pc.reverse_embedding(x_output)

        emb = model(x_real).cpu().tolist()
        emb_conv = model(x_pred_conv).cpu().tolist()
        emb_t = idx2emb[i_trg]

        print('Similarity before transfer:', np.dot(emb, emb_t)[0])
        print('Similarity after transfer:', np.dot(emb_conv, emb_t)[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--dim_emb', type=int, default=128)
    parser.add_argument('--f0_max', type=int, default=1100)
    parser.add_argument('--f0_min', type=int, default=50)
    parser.add_argument('--eng_bin', type=int, default=128)
    parser.add_argument('--eng_max', type=int, default=1)
    parser.add_argument('--eng_min', type=int, default=1e-4)

    # Pitch conversion model configuration.
    parser.add_argument('--chs_grp', type=int, default=16)
    parser.add_argument('--dim_enc', type=int, default=128)
    parser.add_argument('--dim_neck', type=int, default=2)
    parser.add_argument('--freq', type=int, default=128)
    parser.add_argument('--dim_dec', type=int, default=128)
    
    # Training configuration.
    parser.add_argument('--input_path', type=str, default='data/test_audio.wav') # path to source audio
    parser.add_argument('--target_singer', type=str, default='opencpop') # target singer name
    parser.add_argument('--train_pkl_path', type=str, default='train.pkl')
    parser.add_argument('--pc_ckpt_path', type=str, default='ckpt_f0/model_ckpt_steps_400000.ckpt')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt_enc_f0/')
    parser.add_argument('--model_name', type=str, default='ResNetSE1D')
    parser.add_argument('--device', type=torch.device, default='cuda')
    parser.add_argument('--rand_seed', type=int, default=2023)
    
    args = parser.parse_args()

    main(args)
