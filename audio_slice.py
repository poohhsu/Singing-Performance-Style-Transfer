import argparse
import os

import librosa
import soundfile
from tqdm import tqdm

from slicer2 import Slicer


def main(args):
    for root, dirs, files in os.walk(args.input_dir):
        for file in tqdm(sorted(files)):
            if file.endswith('.wav'):
                audio, sr = librosa.load(os.path.join(root, file), sr=None, mono=False)
                if audio.shape[0] > sr * 0.5:
                    slicer = Slicer(
                        sr=sr,
                        threshold=-30,
                        min_length=5000,
                        min_interval=300,
                        hop_size=10,
                        max_sil_kept=500
                    )
                    chunks = slicer.slice(audio)

                    if not os.path.exists(args.output_dir):
                        os.makedirs(args.output_dir)

                    for i, chunk in enumerate(tqdm(chunks)):
                        if len(chunk.shape) > 1:
                            chunk = chunk.T # Swap axes if the audio is stereo.
                        soundfile.write(os.path.join(args.output_dir, f'{root.split("/")[-1]}_{i}.wav'), chunk, sr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/TONAS/') # path to TONAS dataset
    parser.add_argument('--output_dir', type=str, default='data/TONAS_slice/')

    args = parser.parse_args()

    main(args)