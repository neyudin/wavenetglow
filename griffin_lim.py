import numpy as np
import scipy
import librosa


class GriffinLim:
    """
    GriffinLim algorithm
    """
    def __init__(self, sample_rate=24000, num_frequencies=1025, frame_length=0.05, frame_shift=0.0125, mel_channels=80,
                 min_frequency=50, max_frequency=12000, ref_db=20, min_db=-100, preemphasis=0.97, num_iter=20):
        """
        Parameters
        ----------
        sample_rate: int
            Audio sample rate
        num_frequencies: int
            Number of freqiencies to extract
        frame_length: float
            STFT frame length (sec)
        frame_shift: float
            STFT frame shift (sec)
        mel_channels: int
            Number of mel-spectrogram channels
        min_frequency: int
            Minimum frequency to analyze
        max_frequency: int
            Maximum frequency to analyze
        ref_db: int
            Mean db value
        min_db: int
            Min db value
        preemphasis: float
            parameter for filter
        power: float

        num_iter: int
            Number of iterations of the griffin_lim algorithm

        """
        self.sample_rate = sample_rate
        self.num_frequencies = num_frequencies
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.mel_channels = mel_channels
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.ref_db = ref_db
        self.min_db = min_db
        self.preemphasis = preemphasis
        self.num_iter = num_iter

        self.hop_length = int(sample_rate * frame_shift)
        self.win_length = int(sample_rate * frame_length)
        self.n_fft = (num_frequencies - 1) * 2

        mel_basis = librosa.filters.mel(sample_rate, self.n_fft, n_mels=mel_channels,
                                        fmin=min_frequency, fmax=max_frequency)
        self.inv_mel_basis = np.linalg.pinv(mel_basis)

    def stft(self, y):
        """
        Parameters
        ----------
        y: ndarray of size T
            Signal

        Returns
        -------
        z: ndarray of size num_frequencies * num_frames
            Fourier coefficients for the signal

        """
        return librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)

    def istft(self, y):
        """
        Parameters
        ----------
        y: ndarray of size num_frequencies * num_frames
            Fourier coefficients for the signal

        Returns
        -------
        z: ndarray of size T
            Signal

        """
        return librosa.istft(y, hop_length=self.hop_length, win_length=self.win_length)

    def mel_to_linear(self, mel_spectrogram):
        """
        Parameters
        ----------
        mel_spectrogram: ndarray of size mel_channels * num_frames
            Mel-spectrogram

        Returns
        -------
        spectrogram: ndarray of size num_frequencies * num_frames
            Spectrogram

        """
        return np.dot(self.inv_mel_basis, mel_spectrogram)

    def inv_normalize(self, s):
        """
        Parameters
        ----------
        s: ndarray

        Returns
        -------
        output: ndarray

        """
        return s * (-self.min_db) + self.min_db

    def db_to_amp(self, x):
        """
        Parameters
        ----------
        x: ndarray

        Returns
        -------
        output: ndarray

        """
        return 10 ** (x / 20)

    def de_emphasis(self, x):
        """
        Parameters
        ----------
        x: ndarray of size T

        Returns
        -------
        output: ndarray of size T

        """
        return scipy.signal.lfilter([1], [1, -self.preemphasis], x)

    def griffin_lim(self, s):
        """
        Parameters
        ----------
        s: ndarray of size num_frequencies * num_frames
            Module of the spectrogram

        Returns
        -------
        output: ndarray of size num_frequencies * num_frames
            Spectrogram

        """
        np.random.seed(42)
        phi = np.random.uniform(low=0., high=2 * np.pi, size=s.shape)
        for i in range(self.num_iter):
            y = self.istft(s * np.exp(1j * phi))
            phi = np.angle(self.stft(y))
        return s * np.exp(1j * phi)

    def inv_melspectrogram(self, mel_spectrogram):
        """
        Parameters
        ----------
        mel_spectrogram: ndarray of size mel_channels * num_frames
            Mel-spectrogram

        Returns
        -------
        output: ndarray of size T
            Signal

        """
        mel_spectrogram = self.inv_normalize(mel_spectrogram)
        s = self.mel_to_linear(self.db_to_amp(mel_spectrogram + self.ref_db))
        return self.de_emphasis(self.istft(self.griffin_lim(s)))
