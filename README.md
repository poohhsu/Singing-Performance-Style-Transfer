# Many-to-many Singing Performance Style Transfer on Pitch and Energy Contours
This is the source code of "Many-to-many Singing Performance Style Transfer on Pitch and Energy Contours".

More resources including style transfer models and objective evaluation models can be found [here](https://drive.google.com/drive/folders/18674Q414w03XZyIxqfhdsC3FzcoxyosA?usp=sharing).


## Environment

### Install packages
```shell
pip install -r requirements.txt
```


## Preparing data
The training datasets include M4Singer, Opencpop, and TONAS. Please download them in advance and place them in the data folder.
```
data
 |
 ├－ m4singer
 |    ├－ Alto-1#newboy
 |    |    ├－ 0000.wav
 |    |    ├－ 0001.wav
 |    |    └── ...
 |    ├－ Alto-1#云烟成雨
 |    |    ├－ 0000.wav
 |    |    ├－ 0001.wav
 |    |    └── ...
 |    └── ...
 |
 ├－ opencpop
 |    └── segments
 |         ├－ wavs
 |         |    ├－ 2001000001.wav
 |         |    ├－ 2001000002.wav
 |         |    └── ...
 |         └── test.txt
 |
 └── TONAS
      ├－ 01-D_AMairena
      |    └── Vocal.wav
      ├－ 02-D_ChanoLobato
      |    └── Vocal.wav
      └── ...
```


## Preprocessing

### Slice TONAS audio
Since TONAS audio is longer and has not been sliced into segments, it needs to be sliced first.
```
python audio_slice.py
```

### Extract pitch and energy
```
python preprocess.py
```

### Generate training metadata
```
python split_f0_pkl.py
```


## Training

### Pitch
```
python train_f0.py
```

### Energy
```
python train_eng.py
```


## Testing

### Pitch
```
python test_f0.py --input_path $1 --target_singer $2
```
- $1: source audio file path
- $2: target singer name (check train.pkl)

### Energy
```
python test_eng.py --input_path $1 --target_singer $2
```
- $1: source audio file path
- $2: target singer name (check train.pkl)


## Objective evaluation

### Pitch
The emb_ResNetSE1D.pkl and ResNetSE1D.ckpt files must be downloaded from the ckpt_enc_f0 folder in the above link.
```
python test_enc_f0.py --input_path $1 --target_singer $2
```
- $1: source audio file path
- $2: target singer name (check train.pkl)

### Energy
The emb_ResNetSE1D.pkl and ResNetSE1D.ckpt files must be downloaded from the ckpt_enc_eng folder in the above link.
```
python test_enc_eng.py --input_path $1 --target_singer $2
```
- $1: source audio file path
- $2: target singer name (check train.pkl)