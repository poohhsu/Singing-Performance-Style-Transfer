import argparse
import glob
import math
import os
import pickle
import random
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
import torchaudio
from tqdm import tqdm

from model_f0 import PC


class f0Dataset(data.Dataset):
    """Dataset class for the f0 dataset."""

    def __init__(self, split, args):
        """Initialize and preprocess the f0 dataset."""
        self.split = split
        self.len_crop = args.len_crop
        self.step = 10
        
        meta = pickle.load(open(args.train_pkl_path, 'rb'))[split]

        dataset = [None] * len(meta)
        for k, sbmt in enumerate(meta):    
            f0s = []
            for j, tmp in enumerate(sbmt):
                if j < 2:  # fill in speaker id and embedding
                    f0s.append(tmp)
                else: # load the f0 contours
                    f0s.append(np.load(tmp).astype('float32'))
            dataset[k] = f0s
            
        self.train_dataset = list(dataset)
        self.num_tokens = len(self.train_dataset)
        
    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.train_dataset 
        list_f0s = dataset[index]
        c_org = list_f0s[1]
        
        # pick random f0 with random crop
        a, left = np.random.randint(2, len(list_f0s)), 0
        tmp = list_f0s[a].copy()
        if tmp.shape[0] < self.len_crop:
            len_pad = self.len_crop - tmp.shape[0]
            f0 = np.pad(tmp, (0, len_pad), 'edge') # constant
        elif tmp.shape[0] > self.len_crop:
            left = np.random.randint(tmp.shape[0] - self.len_crop)
            f0 = tmp[left: left + self.len_crop]
        else:
            f0 = tmp
        l = min(tmp.shape[0], f0.shape[0])

        # randomly transpose f0
        if self.split == 'train':
            interval = np.random.randint(
                np.ceil(np.log2(args.f0_min / np.min(f0)) * 12),
                np.floor(np.log2(args.f0_max / np.max(f0)) * 12) + 1
            )
            f0 *= 2 ** (interval / 12)
        
        return f0, c_org, l

    def __len__(self):
        """Return the number of singers."""
        return self.num_tokens
    

def get_loader(split, args):
    """Build and return a data loader."""
    
    dataset = f0Dataset(split, args)
    
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  drop_last=True if split == 'train' else False,
                                  worker_init_fn=worker_init_fn)
    return data_loader


def main(args):
    random.seed(args.rand_seed)  
    np.random.seed(args.rand_seed)  
    torch.manual_seed(args.rand_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.rand_seed)
        torch.cuda.manual_seed_all(args.rand_seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    train_loader = get_loader('train', args)
    valid_loader = get_loader('valid', args)

    model = PC(args).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # restore trainer state and model if there is a weight for this experiment
    last_steps = -1
    last_ckpt_name = None
    global_step = 0

    # find last epoch
    checkpoints = os.listdir(args.ckpt_dir)
    for name in checkpoints:
        if '.ckpt' in name and 'steps_' in name:
            steps = name.split('steps_')[1]
            steps = int(re.sub('[^0-9]', '', steps)) - 1

            if steps > last_steps:
                last_steps = steps
                last_ckpt_name = name

    # restore last checkpoint
    if last_ckpt_name is not None:
        checkpoint = torch.load(os.path.join(args.ckpt_dir, last_ckpt_name))
        model.load_state_dict(checkpoint['state_dict'])

        global_step = checkpoint['global_step']
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    # skip the fetched data 
    pbar = tqdm(total=global_step, ncols=0, desc='Train', unit=' step', position=0)
    for i in range(global_step):
        # Fetch data.
        try:
            x_real, c_org, x_len = next(data_iter)
        except:
            data_iter = iter(train_loader)
            x_real, c_org, x_len = next(data_iter)

        pbar.update()
        pbar.set_postfix(step=i + 1)
    pbar.close()

    # Start training.
    total_loss = [0] * 6
    wav2spec = torchaudio.transforms.Spectrogram(n_fft=80, win_length=80, hop_length=20, power=2).to(args.device)
    pbar = tqdm(total=args.num_iters - global_step, ncols=0, desc='Train', unit=' step', position=0)
    for i in range(global_step, args.num_iters):
        # Fetch data.
        try:
            x_real, c_org, x_len = next(data_iter)
        except:
            data_iter = iter(train_loader)
            x_real, c_org, x_len = next(data_iter)
            
        x_real = x_real.to(args.device)[:, :max(x_len)]
        c_org = c_org.to(args.device)
        
        model.train()
        x_input, x_output = model(x_real, c_org)

        # Identity mapping loss.
        loss_id = F.binary_cross_entropy(x_output, x_input)

        # Pitch reconstruction loss.
        x_pred = model.reverse_embedding(x_output)
        loss_pr = F.mse_loss(
            (x_pred / args.f0_min).log2() / math.log2(args.f0_max / args.f0_min),
            (x_real / args.f0_min).log2() / math.log2(args.f0_max / args.f0_min)
        ).sqrt()

        # Fourier transform loss.
        win = torch.special.sinc(torch.arange(-1, 1.02, 0.02)).expand(x_real.shape[0], -1).to(args.device)
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

        # Backward and optimize.
        loss = args.lambda_id * loss_id + args.lambda_pr * loss_pr + args.lambda_ft * loss_ft + args.lambda_ext * loss_ext + args.lambda_sv * loss_sv
        total_loss[0] += loss
        total_loss[1] += loss_id
        total_loss[2] += loss_pr
        total_loss[3] += loss_ft
        total_loss[4] += loss_ext
        total_loss[5] += loss_sv

        loss = loss / args.accumulation_steps
        loss.backward()
        if (i + 1) % args.accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()
            optimizer.zero_grad()

        # Print out training information.
        pbar.update()
        pbar.set_postfix(
            loss=total_loss[0].item() / (i % args.val_check_interval + 1),
            step=i + 1,
        )

        if (i + 1) % args.val_check_interval == 0:
            with open(os.path.join(args.ckpt_dir, 'train_loss.txt'), 'a') as f:
                f.write(' '.join([str(total_loss[j].item() / args.val_check_interval) for j in range(len(total_loss))]) + '\n')

            print()
            total_loss = [0] * 6
            pbar_2 = tqdm(total=len(valid_loader), ncols=0, desc='Valid', unit=' step', position=0)
            model.eval()
            with torch.no_grad():
                for j, (x_real, c_org, x_len) in enumerate(valid_loader):
                    x_real = x_real.to(args.device)[:, :max(x_len)]
                    c_org = c_org.to(args.device)
        
                    x_input, x_output = model(x_real, c_org)

                    # Identity mapping loss.
                    loss_id = F.binary_cross_entropy(x_output, x_input)

                    # Pitch reconstruction loss.
                    x_pred = model.reverse_embedding(x_output)
                    loss_pr = F.mse_loss(
                        (x_pred / args.f0_min).log2() / math.log2(args.f0_max / args.f0_min),
                        (x_real / args.f0_min).log2() / math.log2(args.f0_max / args.f0_min)
                    ).sqrt()

                    # Fourier transform loss.
                    win = torch.special.sinc(torch.arange(-1, 1.02, 0.02)).expand(x_real.shape[0], -1).to(args.device)
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
                    total_loss[0] += loss
                    total_loss[1] += loss_id
                    total_loss[2] += loss_pr
                    total_loss[3] += loss_ft
                    total_loss[4] += loss_ext
                    total_loss[5] += loss_sv

                    pbar_2.update()
                    pbar_2.set_postfix(
                        loss=total_loss[0].item() / (j + 1),
                    )
                pbar_2.close()

            checkpoint = {
                'state_dict': model.state_dict(),
                'global_step': i + 1,
                'optimizer_state': optimizer.state_dict(),
            }
            torch.save(checkpoint, f'{args.ckpt_dir}/model_ckpt_steps_{i + 1}.ckpt')

            for old_ckpt in sorted(glob.glob(f'{args.ckpt_dir}/model_ckpt_steps_*.ckpt'),
                                   key=lambda x: -int(re.findall('.*steps\_(\d+)\.ckpt', x)[0]))[args.num_ckpt_keep:]:
                if int(re.findall('.*steps\_(\d+)\.ckpt', old_ckpt)[0]) % 10000:
                    os.remove(old_ckpt)
                    print(f'Delete ckpt: {os.path.basename(old_ckpt)}')

            with open(os.path.join(args.ckpt_dir, 'val_loss.txt'), 'a') as f:
                f.write(' '.join([str(total_loss[j].item() / len(valid_loader)) for j in range(len(total_loss))]) + '\n')
            total_loss = [0] * 6
    pbar.close()


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
    parser.add_argument('--ckpt_dir', type=str, default='ckpt_f0/')
    parser.add_argument('--train_pkl_path', type=str, default='train.pkl')
    parser.add_argument('--device', type=torch.device, default='cuda')
    parser.add_argument('--rand_seed', type=int, default=2023)
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--num_iters', type=int, default=400000, help='number of total iterations')
    parser.add_argument('--len_crop', type=int, default=3453, help='dataloader output sequence length') # maximum length of training data
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Miscellaneous.
    parser.add_argument('--val_check_interval', type=int, default=1000)
    parser.add_argument('--num_ckpt_keep', type=int, default=11)

    args = parser.parse_args()

    main(args)
