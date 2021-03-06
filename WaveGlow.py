import torch
import torch.nn as nn
import torch.nn.functional as F

from WN import WN


class InvertConv(nn.Module):
    """1x1 Invertible Convolution

    Attributes
    ----------
    conv : nn.Conv1d
        Convolutional filter

    """
    
    def __init__(self, num_channels):
        """
        Parameters
        ----------
        num_channels : int
            Number of x channels
        
        """
        super(InvertConv, self).__init__()
        
        self.conv = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=1,
                              stride=1, padding=0,
                              bias=False)
        
        weight = torch.qr(torch.randn(num_channels, num_channels))[0]
        if torch.det(weight) < 0:
            weight[:, 0] = -weight[:, 0]
        self.conv.weight.data = weight.view(num_channels, num_channels, 1)

    def forward(self, x, reverse=False):
        """
        Parameters
        ----------
        x : FloatTensor of size batch_size * num_channels * T
            Input tensor
        reverse : bool
            True if inference, False if training
        
        Returns
        ----------
        x' : FloatTensor of size batch_size * num_channels * T
            Output tensor
        logdet_W : FloatTensor of size 1, optional
            Only provided if 'reverse' is False
        
        """
        if reverse:
            return F.conv1d(x, weight=self.conv.weight.squeeze().inverse().unsqueeze(2),
                            bias=None, stride=1, padding=0)
        else:
            return self.conv(x), torch.logdet(self.conv.weight.squeeze()) * x.size(0) * x.size(2)


class AffineCouplingLayer(nn.Module):
    """Affine Coupling Layer

    Attributes
    ----------
    wn : WaveNet

    """
    
    def __init__(self, **kwargs):
        """Affine Coupling Layer
        
        Parameters
        ----------
        **kwargs
            Parameters for WaveNet
            
        """
        super(AffineCouplingLayer, self).__init__()
        self.wn = WN(**kwargs)

    def forward(self, x, spect, reverse=False):
        """
        Parameters
        ----------
        x : FloatTensor of size batch_size * num_channels * T
            Input tensor
        spect : FloatTensor of size batch_size * mel_channels * T
            Upsampled mel-spectrogram
        reverse : bool
            True if inference, False if training
        
        Returns
        ----------
        x' : FloatTensor of size batch_size * num_channels * T
            Output tensor
        log_s : FloatTensor of size batch_size * (num_channels // 2) * T, optional
            Only provided if 'reverse' is False
        """
        n_half = x.size(1) // 2
        x_a = x[:, :n_half, :]
        x_b = x[:, n_half:, :]
        output = self.wn(x_a, spect)
        log_s, t = output[:, :n_half, :], output[:, n_half:, :]
        if reverse:
            x_b = (x_b - t) / log_s.exp()
            return torch.cat([x_a, x_b], dim=1)
        else:
            x_b = x_b * log_s.exp() + t
            return torch.cat([x_a, x_b], dim=1), log_s


class WaveGlow(nn.Module):
    """
    WaveGlow network
    """
    def __init__(self, num_blocks, num_channels, mel_channels, early_channels, early_every,
                 **kwargs):
        """
        Parameters
        ----------
        num_blocks : int
            Number of blocks
        num_channels : int
            Number of x channels
        mel_channels : int
            Number of mel-spectrogram channels
        early_channels : int
            Number of channels in early outputs
        early_every : int
            Number of blocks for early outputs
        **kwargs
            Parameters for WaveNet
        
        """
        super(WaveGlow, self).__init__()
        
        self.num_blocks = num_blocks
        self.mel_channels = mel_channels
        self.num_channels = num_channels
        self.early_channels = early_channels
        self.early_every = early_every
        
        # spect upsampling
        self.spect_upsample = nn.ConvTranspose1d(in_channels=self.mel_channels,
                                                 out_channels=self.mel_channels,
                                                 kernel_size=1024, stride=256)
        
        # blocks
        self.invert_conv = nn.ModuleList()
        self.affine_coupling = nn.ModuleList()
        remaining_channels = num_channels
        for i in range(self.num_blocks):
            if i > 0 and i % self.early_every == 0:
                remaining_channels -= self.early_channels
            self.invert_conv.append(InvertConv(remaining_channels))
            self.affine_coupling.append(AffineCouplingLayer(
                num_channels=remaining_channels // 2,
                mel_channels=self.mel_channels * self.num_channels, **kwargs))
        self.num_channels_last = remaining_channels

    def forward(self, x, spect):
        """
        Parameters
        ----------
        x : FloatTensor of size batch_size * audio_len
            Audio sample
        spect : FloatTensor of size batch_size * mel_channels * mel_frames
            Mel-spectrogram
        
        Returns
        ----------
        z : FloatTensor of size batch_size * num_channels * T
            Latent variables
        log_s_list : List of FloatTensors of size batch_size * (num_channels // 2) * T
        logdet_w_list : List of FloatTensors of size 1
        """
        # upsample spect
        spect = self.spect_upsample(spect)
        assert spect.size(2) >= x.size(1)
        if spect.size(2) > x.size(1):
            spect = spect[:, :, :x.size(1)]
        
        # reshape tensors
        x = x.unfold(dimension=1, size=self.num_channels, step=self.num_channels).permute(0, 2, 1)
        spect = spect.unfold(dimension=2, size=self.num_channels,
                             step=self.num_channels).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)
        
        # forward pass
        outputs = []
        log_s_list = []
        logdet_w_list = []
        for i in range(self.num_blocks):
            if i > 0 and i % self.early_every == 0:
                outputs.append(x[:, :self.early_channels, :])
                x = x[:, self.early_channels:, :]
            x, logdet_w = self.invert_conv[i](x)
            x, log_s = self.affine_coupling[i](x, spect)
            log_s_list.append(log_s)
            logdet_w_list.append(logdet_w)
        outputs.append(x)
        
        return torch.cat(outputs, dim=1), log_s_list, logdet_w_list
        
    def infer(self, spect, sigma=1.0):
        """
        Parameters
        ----------
        spect : FloatTensor of size batch_size * mel_channels * mel_frames
            Mel-spectrogram
        sigma : float scalar
            Standard deviation of latent variables
        
        Returns
        ----------
        x : FloatTensor of size batch_size * audio_len
            Audio sample
        """
        spect = self.spect_upsample(spect)
        # в статье говорят, что нужно избавиться от артефактов
        spect = spect[:, :, :-(self.spect_upsample.kernel_size[0] - self.spect_upsample.stride[0])]

        spect = spect.unfold(2, self.num_channels, self.num_channels).permute(0, 2, 1, 3)
        spect = spect.contiguous().view(spect.size(0), spect.size(1), -1).permute(0, 2, 1)

        x = sigma * torch.randn(spect.size(0), self.num_channels_last, spect.size(2),
                                device=spect.device)

        for i in range(self.num_blocks - 1, -1, -1):
            x = self.affine_coupling[i](x, spect, reverse=True)
            x = self.invert_conv[i](x, reverse=True)

            if i > 0 and i % self.early_every == 0:
                z = sigma * torch.randn(spect.size(0), self.early_channels, spect.size(2),
                                        device=spect.device)
                x = torch.cat([z, x], dim=1)

        return x.permute(0, 2, 1).contiguous().view(x.size(0), -1).data
    
    def compute_loss(self, x, spect, sigma=1.0):
        """
        Parameters
        ----------
        x : FloatTensor of size batch_size * audio_len
            Audio sample
        spect : FloatTensor of size batch_size * mel_channels * mel_frames
            Mel-spectrogram
        sigma : float scalar
            Standard deviation of latent variables
        
        Returns
        ----------
        FloatTensor of size 1
            Loss value
        """
        z, log_s_list, logdet_w_list = self.forward(x, spect)
        loss = torch.sum(z ** 2) / (2 * sigma ** 2)
        for i in range(self.num_blocks):
            loss -= torch.sum(log_s_list[i])
            loss -= logdet_w_list[i]
        
        return loss / (z.size(0) * z.size(1) * z.size(2))
