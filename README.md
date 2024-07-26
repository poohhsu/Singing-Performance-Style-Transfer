# Any-to-many Singing Style Conversion
This is the source code of "Any-to-many Singing Style Conversion".

More resources can be found [here](https://drive.google.com/drive/folders/18674Q414w03XZyIxqfhdsC3FzcoxyosA?usp=sharing).


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
 |    |    ├－ 0000.mid
 |    |    ├－ 0000.TextGrid
 |    |    ├－ 0000.wav
 |    |    └── ...
 |    └── ...
 |
 ├－ opencpop
 |    ├－ midis
 |    |    └── ...
 |    └── ...
 |
 └── TONAS
      ├－ 01-D_AMairena
      |    ├－ gt.txt
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