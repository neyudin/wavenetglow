import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import get_window
from librosa.util import pad_center
from numpy.fft import fft
from librosa.filters import mel


class PytorchSTFT(nn.Module):
    """Pytorch implementation of the Short-time Fourier transform (STFT).

    This class contains only necessary aspects of the STFT to transform input signal into
    multidimensional sequence of magnitudes.

    Attributes
    ----------
    n_fft : int > 0 [scalar]
        Number of components in the fast Fourier transform (FFT).
    hop_len : int > 0 [scalar]
        Number audio of frames between STFT columns.
    win_len : int > 0 [scalar]
        Each frame of audio is windowed by `hann`.
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`. `n_fft` >= `win_len`.
    self.cutoff_freq : int > 0 [scalar]
        Cutoff frequency of the real-valued signal.
        Used to tackle the symmetricity of the real-valued signal.
    basis : torch.Tensor [shape=(`n_fft`, `n_fft`)]
        Fourier basis.

    """

    def __init__(self, n_fft, hop_len, win_len):
        """
        Parameters
        ----------
        n_fft : int > 0 [scalar]
            Number of components in the fast Fourier transform (FFT).
        hop_len : int > 0 [scalar]
            Number audio of frames between STFT columns.
        win_len : int > 0 [scalar]
            Each frame of audio is windowed by `hann`.
            The window will be of length `win_length` and then padded
            with zeros to match `n_fft`. `n_fft` >= `win_len`.

        """
        super(PytorchSTFT, self).__init__()

        self.n_fft = int(n_fft)
        self.hop_len = int(hop_len)
        self.win_len = int(win_len)

        assert (self.n_fft >= self.win_len)

        self.cutoff_freq = self.n_fft // 2 + 1

        fft_basis = fft(np.eye(self.n_fft))
        fft_basis = np.vstack([np.real(fft_basis[:self.cutoff_freq, :]),
                               np.imag(fft_basis[:self.cutoff_freq, :])])
        fft_basis = torch.Tensor(fft_basis[:, np.newaxis, :])

        fft_win = torch.Tensor(pad_center(
            data=get_window(
                window='hann',
                Nx=self.win_len),
            size=self.n_fft).astype(np.float32))

        self.basis = fft_basis * fft_win

    def forward(self, input):
        """
        Parameters
        ----------
        input : torch.Tensor [shape=(batch_size, timestamps)]
            Batch of the input signals.

        Returns
        -------
        torch.Tensor [shape=(batch_size, 1 + `n_fft` // 2, timestamps)]
            Magnitude of the STFT.

        """
        with torch.no_grad():
            input = F.pad(input=input.view(input.shape[0], 1, 1, input.shape[1]),
                          pad=[self.n_fft // 2, self.n_fft // 2, 0, 0],
                          mode='reflect').squeeze(1)
            input = F.conv1d(input=input, weight=self.basis, stride=self.hop_len)

        return torch.sqrt(input[:, :self.cutoff_freq, :]**2 + input[:, self.cutoff_freq:, :]**2)


class PytorchMelSTFT(nn.Module):
    """Combines FFT bins in STFT into Mel-frequency bins

    This class contains only necessary aspects of the STFT to transform input signal into
    multidimensional sequence of magnitudes in Mel-frequency scale.

    Attributes
    ----------
    sr : int > 0 [scalar]
        Sampling rate of the incoming signal.
    n_fft : int > 0 [scalar]
        Number of components in the fast Fourier transform (FFT).
    fmin : float >=0 [scalar]
        Lowest frequency (in Hz).
    fmax : float >= 0 [scalar]
        Highest frequency (in Hz).
    hop_len : int > 0 [scalar]
        Number audio of frames between STFT columns.
    win_len : int > 0 [scalar]
        Each frame of audio is windowed by `hann`.
        The window will be of length `win_length` and then padded
        with zeros to match `n_fft`. `n_fft` >= `win_len`.
    n_mels : int > 0 [scalar], default -- 80
        Number of Mel bands to generate.
    stft_transformer : nn.Module, callable
        Callable nn.Module STFT transformer.
    basis : torch.Tensor [shape=(`n_mels`, 1 + `n_fft` // 2)]
        Mel transform matrix.

    """

    def __init__(self, sr, n_fft, fmin, fmax, hop_len, win_len, n_mels=80):
        """
        Parameters
        ----------
        sr : int > 0 [scalar]
            Sampling rate of the incoming signal.
        n_fft : int > 0 [scalar]
            Number of components in the fast Fourier transform (FFT).
        fmin : float >=0 [scalar]
            Lowest frequency (in Hz).
        fmax : float >= 0 [scalar]
            Highest frequency (in Hz).
        hop_len : int > 0 [scalar]
            Number audio of frames between STFT columns.
        win_len : int > 0 [scalar]
            Each frame of audio is windowed by `hann`.
            The window will be of length `win_length` and then padded
            with zeros to match `n_fft`. `n_fft` >= `win_len`.
        n_mels : int > 0 [scalar], default -- 80
            Number of Mel bands to generate.

        """
        super(PytorchMelSTFT, self).__init__()

        self.sr = sr
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.hop_len = hop_len
        self.win_len = win_len
        self.n_mels = n_mels

        self.stft_transformer = PytorchSTFT(
            n_fft=self.n_fft,
            hop_len=self.hop_len,
            win_len=self.win_len)

        self.basis = torch.Tensor(mel(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax).astype(np.float32))

    def forward(self, input):
        """
        Parameters
        ----------
        input : torch.Tensor [shape=(batch_size, timestamps)]
            Batch of the input signals in range [-1, 1].

        Returns
        -------
        torch.Tensor [shape=(batch_size, `n_mels`, timestamps)]
            Magnitude of the STFT in Mel-frequency scale.

        """
        assert (torch.min(input) >= -1)
        assert (torch.max(input) <= 1)

        with torch.no_grad():
            input = self.stft_transformer(input)
            input = torch.log(torch.clamp(torch.matmul(self.basis, input), min=1e-5))

        return input
