import torch
import torch.nn.functional as F
from torch.utils import data
from scipy.io.wavfile import read
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sound_utils import PytorchMelSTFT


def get_wav_list(directory):
    """Extracts list of .wav audio files' paths from the given directory.

    Parameters
    ----------
    directory : str
        Path to the directory.

    Returns
    -------
    list
        List of the paths of .wav audio files.

    """
    return [file.rstrip() for file in glob(directory + '/*.wav')]


def wav2tensor(path):
    """Loads .wav audio file into Pytorch Tensor.

    Parameters
    ----------
    path : str
        Path to the audio file.

    Returns
    -------
    int
        Sampling rate.
    torch.Tensor [shape=(n,)]
        1d audio signal with n length.

    """
    result = read(path)
    return result[0], torch.Tensor(result[1].astype(np.float32))


def get_train_test_paths(paths, test_size=0.3, random_state=517, shuffle=True):
    """Divides list of file names into train and test sets.

    Parameters
    ----------
    paths : list
        List of file names.
    test_size : float [scalar], default -- 0.3
        Test set fraction in range (0, 1).
    random_state : int [scalar], default -- 517
        Seed for pseudo random numbers generator.
    shuffle : bool, default -- True
        Whether shuffle the input data set `paths`.

    Returns
    -------
    list
        Train paths.
    list
        Test paths.

    """
    return train_test_split(paths, test_size=test_size, random_state=random_state, shuffle=shuffle)


class MelSpectrogramDataset(data.Dataset):
    """Loads audio files and converts it into Mel-spectrograms.

    The data set representation to work with Pytorch models.

    Attributes
    ----------
    paths : list
        List of .wav file names.
    sr : int > 0 [scalar]
        Sampling rate of the incoming signal.
    n_fft : int > 0 [scalar]
        Number of components in the fast Fourier transform (FFT).
    fmin : float >= 0 [scalar]
        Lowest frequency (in Hz).
    fmax : float >= 0 [scalar]
        Highest frequency (in Hz).
    hop_len : int > 0 [scalar]
        Number audio of frames between STFT columns.
    win_len : int > 0 [scalar]
        Each frame of audio is windowed by `hann`.
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`. `n_fft` >= `win_len`.
    seg_len : int > 0 [scalar]
        Length of the sampled audio.
    seed : int [scalar], default -- 42
        Seed for pseudo random numbers generator.
    shuffle : bool, default -- True
        Whether shuffle the input data set `paths`.
    max_wav_val : int > 0 [scalar], default -- 32768
        Maximal amplitude of the .wav audio, used to map signal to [-1, 1] range.
    wav2mel_transformer : callable nn.Module
        Transforms input audio into Mel-spectrogram.

    """

    def __init__(self, data_dir, sr, n_fft, fmin, fmax, hop_len, win_len, seg_len, seed=42, shuffle=True,
                 max_wav_val=32768):
        """
        Parameters
        ----------
        data_dir : str
            Path to the directory with .wav files.
        sr : int > 0 [scalar]
            Sampling rate of the incoming signal.
        n_fft : int > 0 [scalar]
            Number of components in the fast Fourier transform (FFT).
        fmin : float >= 0 [scalar]
            Lowest frequency (in Hz).
        fmax : float >= 0 [scalar]
            Highest frequency (in Hz).
        hop_len : int > 0 [scalar]
            Number audio of frames between STFT columns.
        win_len : int > 0 [scalar]
            Each frame of audio is windowed by `hann`.
            The window will be of length `win_length` and then padded
            with zeros to match `n_fft`. `n_fft` >= `win_len`.
        seg_len : int > 0 [scalar]
            Length of the sampled audio.
        seed : int [scalar], default -- 42
            Seed for pseudo random numbers generator.
        shuffle : bool, default -- True
            Whether shuffle the input data set `paths`.
        max_wav_val : int > 0 [scalar], default -- 32768
            Maximal amplitude of the .wav audio, used to map signal to [-1, 1] range.

        """
        self.seed = seed
        np.random.seed(self.seed)
        self.paths = sorted(get_wav_list(data_dir))
        self.sr = sr
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.hop_len = hop_len
        self.win_len = win_len
        self.seg_len = seg_len
        self.shuffle = shuffle
        self.max_wav_val = max_wav_val

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
        """
        Parameters
        ----------
        index : int
            Number of the element in this data set.

        Returns
        -------
        torch.Tensor
            Mel-spectrogram.
        torch.Tensor
            Audio in [-1, 1] range.

        """
        sr, sound = wav2tensor(self.paths[index])
        sound /= self.max_wav_val

        assert (self.sr == sr)

        if self.seg_len > 0:
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
        """
        Returns
        -------
        int
            Data set length.

        """
        return len(self.paths)
