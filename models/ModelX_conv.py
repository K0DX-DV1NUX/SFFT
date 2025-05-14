import torch
import torch.nn as nn
import torch.nn.functional as F



# === Trend Extraction via Moving Average ===
class TrendExtractor(nn.Module):
    """
    Extracts trend from the time series using average pooling,
    with front-padding to preserve alignment.
    """
    def __init__(self, kernel_size: int, stride: int):
        super(TrendExtractor, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, 
                                stride=stride, 
                                padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, channels, seq_len]
        # Front-pad with the first value to preserve shape
        front = x[:, :, 0:1].repeat(1, 1, self.kernel_size - 1)
        x = torch.cat([front, x], dim=-1)
        return self.avg(x)  # [batch_size, channels, seq_len]    
    
    # # x: [batch_size, channels, seq_len]
    # def forward(self, x):
    #     #padding on front end of time series
    #     front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1))

    #     # padding on the both ends of time series
    #     #front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
    #     #end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
    #     #x = torch.cat([front, x, end], dim=1)

    #     x = torch.cat([front, x], dim=-1)
    #     x = self.avg(x)
    #     return x # [batch_size, channels, seq_len]
    
    
# === Seasonal Pattern Extraction via Depthwise Convolution ===
class SeasonalExtractor(nn.Module):
    """
    Extracts seasonal patterns via circular depthwise convolution
    with symmetric kernel constraint.
    """

    def __init__(self, kernel_size: int, channels: int):
        super(SeasonalExtractor, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.channels = channels

        self.season = nn.Conv1d(in_channels=channels,
                                out_channels=channels,
                                kernel_size=self.kernel_size,
                                stride=self.stride,
                                groups=channels,
                                padding=0,
                                bias=False)

        # Initialize weights symmetrically
        with torch.no_grad():
            for c in range(channels):
                w = torch.randn(self.kernel_size)
                sym_w = 0.5 * (w + torch.flip(w, dims=[0]))  # symmetric kernel
                self.season.weight[c, 0] = sym_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, channels, seq_len]
        left_pad = self.kernel_size // 2
        right_pad = self.kernel_size - left_pad - 1
        x_padded = F.pad(x, (left_pad, right_pad), mode='circular')
        return F.tanh(self.season(x_padded))

    def symmetry_regularizer(self) -> torch.Tensor:
        """
        Computes regularization loss to enforce symmetry of convolution kernels.
        """
        loss = 0.0
        kernel = self.season.weight  # shape: [C, 1, K]
        flipped = torch.flip(kernel, dims=[-1])  # flip over kernel dimension
        diff = kernel - flipped
        loss = 0.5 * torch.sum(diff ** 2)  # sum over all elements
        return loss


# === One-Layer Encoder Block: Trend + Multiple Seasonals ===
class DenoisingBlock(nn.Module):
    """
    A single denoising block that extracts trend and multiple seasonal patterns.
    """

    def __init__(self, trend_kernel_size: int, seasonal_kernel_size: int, channels: int, num_seasonals: int):
        super(DenoisingBlock, self).__init__()
        self.trend = TrendExtractor(kernel_size=trend_kernel_size, stride=1)
        self.multi_seasonals = nn.ModuleList([
            SeasonalExtractor(kernel_size=seasonal_kernel_size, channels=channels)
            for _ in range(num_seasonals)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_T = self.trend(x)
        x_rem = x - x_T
        for s in self.multi_seasonals:
            x_S = s(x_rem)
            x_rem = x_rem - x_S
        return x - x_rem  # Reconstruct denoised signal

    def symmetry_regularizer(self) -> torch.Tensor:
        return sum(s.symmetry_regularizer() for s in self.multi_seasonals)

# === Stacked Denoising Blocks ===
class DenoisingStack(nn.Module):
    """
    Stacked DenoisingBlocks to iteratively refine signal denoising.
    """

    def __init__(self, num_blocks: int, trend_kernel_size: int, seasonal_kernel_size: int, num_seasonals: int, channels: int):
        super(DenoisingStack, self).__init__()
        self.blocks = nn.ModuleList([
            DenoisingBlock(trend_kernel_size=trend_kernel_size,
                           seasonal_kernel_size=seasonal_kernel_size,
                           channels=channels,
                           num_seasonals=num_seasonals)
            for _ in range(num_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

    def symmetry_regularizer(self) -> torch.Tensor:
        return sum(block.symmetry_regularizer() for block in self.blocks)



class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  # Input sequence length
        self.pred_len = configs.pred_len  # Prediction horizon
        self.channels = configs.enc_in  # Number of input channels (features)
        
        # Model architecture hyperparameters
        encoder_depth = 3
        trend_kernel_size = 25
        num_seasonals = 3

        # Denoising module
        self.encoder = DenoisingStack(num_blocks=encoder_depth,
                                      trend_kernel_size=trend_kernel_size,
                                      seasonal_kernel_size=self.seq_len,
                                      num_seasonals=num_seasonals,
                                      channels=self.channels)

        # Final prediction layer: maps denoised seq to future horizon
        self.pred = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, channels]
        x = x.permute(0, 2, 1)  # -> [B, C, L]
        seq_mean = torch.mean(x, dim=-1, keepdim=True)
        x = x - seq_mean

        x = self.encoder(x)
        out = self.pred(x)  # [B, C, pred_len]
        out = out.permute(0, 2, 1) + seq_mean  # restore mean
        return out  # [B, pred_len, C]

    def symmetry_regularizer(self) -> torch.Tensor:
        return self.encoder.symmetry_regularizer()