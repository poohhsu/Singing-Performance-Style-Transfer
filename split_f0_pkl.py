import argparse
from collections import defaultdict
import os
import pickle

import numpy as np


singer2idx = {
    'm4singer_Alto-1': 200,
    'm4singer_Alto-2': 201,
    'm4singer_Alto-3': 202,
    'm4singer_Alto-4': 203,
    'm4singer_Alto-5': 204,
    'm4singer_Alto-6': 205,
    'm4singer_Alto-7': 206,
    'm4singer_Bass-1': 207,
    'm4singer_Bass-2': 208,
    'm4singer_Bass-3': 209,
    'm4singer_Soprano-1': 210,
    'm4singer_Soprano-2': 211,
    'm4singer_Soprano-3': 212,
    'm4singer_Tenor-1': 213,
    'm4singer_Tenor-2': 214,
    'm4singer_Tenor-3': 215,
    'm4singer_Tenor-4': 216,
    'm4singer_Tenor-5': 217,
    'm4singer_Tenor-6': 218,
    'm4singer_Tenor-7': 219,
}


def main(args):
    opencpop_test_song = set()
    with open(args.opencpop_test) as f:
        for line in f.readlines():
            opencpop_test_song.add(line[:4])

    train_singers, valid_singers, test_singers = [], [], []
    for root, dirs, files in os.walk(args.f0_dir):
        singer = root.split('/')[-1]
        if '/m4singer' in root:
            idx = singer2idx[singer]
        elif '/opencpop' in root:
            idx = 255
        elif '/TONAS' in root:
            idx = 252
        # elif '/OpenSinger' in root: # for testing
        #     idx = int(root.split('_')[-1]) 
        else:
            continue

        print('Processing singer: %s' % singer)

        total = 0
        songs = defaultdict(list)
        for f0_file in sorted(files):
            if f0_file.endswith('.npy'):
                if '/m4singer' in root:
                    song = f0_file.split('_')[0]
                elif '/opencpop' in root:
                    song = f0_file[:4]
                elif '/TONAS' in root:
                    song = f0_file.split('_')[0]
                # elif '/OpenSinger' in root: # for testing
                #     song = f0_file.split('_')[1]
                else:
                    continue

                f0_path = os.path.join(args.f0_dir, singer, f0_file)
                if 100 < np.load(f0_path).shape[0] < 4000: # skip too short or too long audios
                    songs[song].append(f0_path)
                    total += 1

        count = 0
        train_segments, valid_segments, test_segments = [singer, np.asarray(idx, np.int32)], [singer, np.asarray(idx, np.int32)], [singer, np.asarray(idx, np.int32)]
        for k in sorted(songs.keys()):
            count += len(songs[k])
            if '/opencpop' in root:
                if k in opencpop_test_song:
                    test_segments += songs[k]
                elif count < total * args.train_ratio:
                    train_segments += songs[k]
                else:
                    valid_segments += songs[k]
            else:
                if count < total * args.train_ratio:
                    train_segments += songs[k]
                elif count < total * (0.5 + args.train_ratio/2) or len(valid_segments) == 2:
                    valid_segments += songs[k]
                else:
                    test_segments += songs[k]
                
        train_singers.append(train_segments)
        valid_singers.append(valid_segments)
        test_singers.append(test_segments)

        print(len(train_segments), len(valid_segments), len(test_segments))
        
    with open('train.pkl', 'wb') as handle:
        pickle.dump({
            'train': sorted(train_singers),
            'valid': sorted(valid_singers),
            'test': sorted(test_singers)
        }, handle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f0_dir', type=str, default='data_f0/')
    parser.add_argument('--opencpop_test', type=str, default='data/opencpop/segments/test.txt') # path to test.txt of Opencpop dataset
    parser.add_argument('--train_ratio', type=float, default=0.8, help='training data ratio')

    args = parser.parse_args()

    main(args)