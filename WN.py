import torch
import torch.nn as nn


class WN(torch.nn.Module):
    """
    Это WaveNet как в статье про WaveGlow
    """
    
    def __init__(self, num_channels, mel_channels, n_layers=8, residual_channels=512, 
                 gate_channels=256, skip_channels=256, pre_channels=128):
        
        """
        Parameters
        ----------
        num_channels : int
            Number of x_a channels
        mel_channels : int
            Number of spectrogram (condition c) channels
        ----------  
        
        Parameters from original paper
        ----------  
        n_layers : int
            The depth of WN (default : 8)
        residual_channels : int
            Number of chanels used by residual connections (default : 512)
        gate_channels : int
            Number of filters and gates channels (default : 256)
        skip_channels : int
            Number of chanels used by skip connections
        pre_channels : int
            Number of channels in final non-linearity
        """
        
        super(WN, self).__init__()
        
        self.n_layers = n_layers
        self.num_channels = num_channels
        self.residual_channels = residual_channels
        self.gate_channels = gate_channels
        self.skip_channels = skip_channels
        self.mel_channels = mel_channels
        
        self.dilations_list = [2**i for i in range(n_layers)]
        
        self.conv_input = nn.Conv1d(in_channels=num_channels, out_channels=residual_channels, kernel_size=1)
        
        self.conv_filter = nn.ModuleList([
            nn.Conv1d(
                in_channels=residual_channels,
                out_channels=gate_channels,
                kernel_size=3,
                dilation=d,
                padding=(2 * d // 2)
            ) for d in self.dilations_list])

        self.conv_gate = nn.ModuleList([
            nn.Conv1d(
                in_channels=residual_channels,
                out_channels=gate_channels,
                kernel_size=3,
                dilation=d,
                padding=(2 * d // 2)
            ) for d in self.dilations_list])
        
        self.conv_mel = nn.ModuleList([
                nn.Conv1d(
                    in_channels=mel_channels,
                    out_channels=gate_channels * 2,
                    kernel_size=1
                ) for _ in range(len(self.dilations_list))])
        
        self.conv_residual = nn.ModuleList([
            nn.Conv1d(
                in_channels=gate_channels,
                out_channels=residual_channels,
                kernel_size=1
            ) for _ in range(len(self.dilations_list) - 1)])
        
        self.conv_skip = nn.ModuleList([
            nn.Conv1d(
                in_channels=gate_channels,
                out_channels=skip_channels,
                kernel_size=1
            ) for _ in range(len(self.dilations_list))])
        
        self.conv_out_1 = nn.Conv1d(
            in_channels=skip_channels,
            out_channels=pre_channels,
            kernel_size=1)
        self.conv_out_2 = nn.Conv1d(
            in_channels=pre_channels,
            out_channels=2 * num_channels,
            kernel_size=1)

    def forward(self, x_a, c):
        """
        Parameters
        ----------
        x_a : FloatTensor of size batch_size * num_channels * T
            Unchangable part of embedding
        c : FloatTensor of size batch_size * mel_channels * T
            Upsampled mel-spectrogram
        """
        assert x_a.size(2) == c.size(2)  # Проверить, что спектрограмме не забыли сделать upsampling
        
        x_acc = 0
        x = self.conv_input(x_a)
        for i in range(len(self.dilations_list)):
            x_filter = self.conv_filter[i](x)
            x_gate = self.conv_gate[i](x)
            c_proj = self.conv_mel[i](c)
            x_filter = x_filter + c_proj[:, :self.gate_channels]
            x_gate = x_gate + c_proj[:, self.gate_channels:]
            x_gate = torch.sigmoid(x_gate)
            x_filter = torch.tanh(x_filter)
            x_filter_gate = x_gate * x_filter
            x_skip = self.conv_skip[i](x_filter_gate)
            if i != len(self.dilations_list) - 1:
                x_res = self.conv_residual[i](x_filter_gate)
                x = x + x_res
                # x = x * 0.5**0.5
            x_acc = x_acc + x_skip
            
        return self.conv_out_2(torch.relu(self.conv_out_1(x_acc)))
