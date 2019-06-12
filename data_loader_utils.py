import torch
import torch.nn.functional as F
from torch.utils import data
from scipy.io.wavfile import read
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sound_utils import PytorchMelSTFT


def get_wav_list(directory):
    return [file.rstrip() for file in glob(directory + '/*.wav')]


def wav2tensor(path):
    result = read(path)
    return result[0], torch.from_numpy(result[1].astype(np.float32))


def get_train_test_paths(paths, test_size=0.3, random_state=517, shuffle=True):
    return train_test_split(paths, test_size=test_size, random_state=random_state, shuffle=shuffle)


class MelSpectrogramDataset(data.Dataset):

    def __init__(self, data_dir, sr, n_fft, fmin, fmax, hop_len, win_len, seg_len, seed=42, shuffle=True,
                 max_wav_val=32768):
        self.paths = get_wav_list(data_dir)
        self.sr = sr
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.hop_len = hop_len
        self.win_len = win_len
        self.seg_len = seg_len
        self.seed = seed
        self.shuffle = shuffle
        self.max_wav_val = max_wav_val

        np.random.seed(self.seed)
        if self.shuffle:
            np.random.shuffle(self.paths)

        self.wav2mel_transformer = PytorchMelSTFT(
            sr=self.sr,
            n_fft=self.n_fft,
            fmin=self.fmin,
            fmax=self.fmax,
            hop_len=self.hop_len,
            win_len=self.win_len)

    def __getitem__(self, index):
        sr, sound = wav2tensor(self.paths[index])
        sound /= self.max_wav_val

        assert (self.sr == sr)

        if sound.shape[0] < self.seg_len:
            sound = F.pad(input=sound,
                          pad=[0, self.seg_len - sound.shape[0]],
                          mode='constant')
        else:
            start_pos = np.random.randint(0, sound.shape[0] - self.seg_len)
            sound = sound[start_pos:start_pos + self.seg_len]

        with torch.no_grad():
            mel = self.wav2mel_transformer(sound.unsqueeze(0)).squeeze(0)

        return mel, sound

    def __len__(self):
        return len(self.paths)
