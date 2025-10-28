import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRank(nn.Module):
    """
    Implements a low-rank approximation layer using two smaller weight matrices (A and B).
    This reduces the number of parameters compared to a full-rank layer.
    """
    def __init__(self, in_features, out_features, rank, bias=True):
        super(LowRank, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.bias = bias

        # Initialize weight matrices A (in_features x rank) and B (rank x out_features)
        wA = torch.empty(self.in_features, rank)
        wB = torch.empty(self.rank, self.out_features)
        self.A = nn.Parameter(nn.init.kaiming_uniform_(wA))
        self.B = nn.Parameter(nn.init.kaiming_uniform_(wB))

        # Initialize bias if required
        if self.bias:
            wb = torch.empty(self.out_features)
            self.b = nn.Parameter(nn.init.uniform_(wb))

    def forward(self, x):
        # Apply low-rank transformation: X * A * B
        out = x @ self.A
        out = out @ self.B
        if self.bias:
            out += self.b  # Add bias if enabled
        return out


class Trend(nn.Module):
    """
    A module to compute the trend component using average pooling over a 1D sequence.

    Args:
    - kernel_size (int): Size of the averaging kernel.
    - stride (int): Stride of the averaging operation.

    Input:
    - x (torch.Tensor): Input tensor of shape [batch_size, channels, seq_len].

    Output:
    - x (torch.Tensor): Trend component tensor of shape [batch_size, channels, seq_len].
    """

    def __init__(self, kernel_size, stride):
        super(Trend, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Pad the beginning of the time series to account for kernel size
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1))
        x = torch.cat([front, x], dim=-1)
        x = self.avg(x)
        return x
    
    
class Seasonal(nn.Module):
    """
    A module to model the seasonal component of a time series using a learned symmetric circulant matrix.

    The circulant matrix is parameterized in the Fourier domain via a learned diagonal in the frequency space.
    The matrix is constructed to be (approximately) symmetric through regularization.

    Args:
    - seq_len (int): Length of the input sequence.
    - f_basis (torch.Tensor): Precomputed Fourier basis of shape (seq_len, seq_len), complex64.

    Input:
    - x (torch.Tensor): Input tensor of shape [batch_size, seq_len].

    Output:
    - x (torch.Tensor): Transformed seasonal component of shape [batch_size, seq_len].
    """

    def __init__(self, seq_len, f_basis):
        super(Seasonal, self).__init__()
        self.seq_len = seq_len
        
        # Register Fourier basis and its inverse (conjugate transpose)
        self.register_buffer('f_modes', f_basis)
        self.register_buffer('f_modes_inv', torch.conj(self.f_modes).T)
        
        # Initialize the diagonal in the frequency domain to parameterize the circulant matrix
        # This is a real-valued vector of length seq_len
        # The real circulant matrix will be enforced via symmetry regularization
        self.diagonal = nn.Parameter(
            nn.init.uniform_(torch.empty(seq_len), a=-1.0, b=1.0),
            requires_grad=True
        )

    def forward(self, x):
        # Construct diagonal matrix in Fourier domain (complex-valued)
        diagonal_mat = torch.diag(self.diagonal).to(torch.complex64)

        # Build the circulant matrix in the time domain:
        # circulant_mat = F^{-1} * diag(diagonal) * F
        circulant_mat = self.f_modes_inv @ diagonal_mat @ self.f_modes

        # Apply the real part of the circulant matrix to the input
        x = x @ circulant_mat.real

        # Apply a nonlinearity (tanh)
        return F.tanh(x)

    @staticmethod
    def _fourier_basis(N):
        """
        Generate the normalized discrete Fourier transform (DFT) matrix F of size (N, N).

        F[j, k] = 1/sqrt(N) * exp(-2 * pi * i * j * k / N)

        Args:
        - N (int): Size of the DFT matrix.

        Returns:
        - F (torch.Tensor): Complex-valued DFT matrix of shape (N, N), dtype=torch.complex64.
        """
        # Generate the row indices (j) and column indices (k)
        j = torch.arange(N).view(-1, 1)  # shape (N, 1)
        k = torch.arange(N).view(1, -1)  # shape (1, N)

        # Compute the complex exponential part
        exponent = -2j * torch.pi * j * k / N

        # Return the normalized DFT matrix
        return (1 / torch.sqrt(torch.tensor(N, dtype=torch.float32))) * torch.exp(exponent)

    def symmetry_regularizer(self):
        """
        Compute a regularization term encouraging the real part of the circulant matrix to be symmetric.

        Returns:
        - reg_term (torch.Tensor): Scalar tensor representing the Frobenius norm of (C - C^T).
        """
        # Construct diagonal matrix in Fourier domain
        diagonal_mat = torch.diag(self.diagonal).to(torch.complex64)

        # Build the circulant matrix in time domain
        circulant_mat = self.f_modes_inv @ diagonal_mat @ self.f_modes

        # Extract the real part
        circulant_mat = circulant_mat.real

        # Compute Frobenius norm of (C - C^T), encouraging symmetry
        return (circulant_mat - circulant_mat.T).norm(p=2)


class DecompBlock(nn.Module):
    """
    A single decomposition block that iteratively removes seasonal and trend components from a time series.

    The block applies:
    1. Multiple learned symmetric circulant operators to extract seasonal components.
    2. An average pooling operation to extract the trend component.

    Args:
    - trend_kernel_size (int): Kernel size for the Trend module.
    - seq_len (int): Length of the input time series.
    - seasons (int): Number of separate Seasonal modules (to capture multiple seasonalities).

    Input:
    - x (torch.Tensor): Input tensor of shape [batch_size, channels, seq_len].

    Output:
    - x (torch.Tensor): Denoised version of the input time series, after removing seasonal and trend components.
    """

    def __init__(self, trend_kernel_size, seq_len, seasons):
        super(DecompBlock, self).__init__()

        # Trend extraction module
        self.trend = Trend(kernel_size=trend_kernel_size, stride=1)

        # Precompute Fourier basis for the given sequence length
        f_modes = Seasonal._fourier_basis(seq_len * seasons)
        seasonal_f_modes = [f_modes[i*seq_len:(i+1)*seq_len, i*seq_len:(i+1)*seq_len] for i in range(seasons)]

        # Build multiple Seasonal modules (one per "seasonality component")
        self.seasonal_list = nn.ModuleList([
            Seasonal(seq_len, seasonal_f_modes[_]) for _ in range(seasons)
        ])

    def forward(self, x):
        # Clone the input for residual calculation
        x_rem = x.detach().clone()

        # Iteratively extract and subtract each seasonal component
        for seasonal in self.seasonal_list:
            x_seasonal = seasonal(x_rem)
            x_rem = x_rem - x_seasonal

        # Extract and subtract the trend component
        x_T = self.trend(x_rem)
        x_rem = x_rem - x_T

        # Return the denoised version of the original input
        return x - x_rem

    def symmetry_regularizer(self):
        """
        Compute the total symmetry regularization term for all seasonal components in this block.

        Returns:
        - reg (torch.Tensor): Scalar tensor representing the total symmetry regularization.
        """
        reg = 0.0
        for seasonal in self.seasonal_list:
            reg += seasonal.symmetry_regularizer()
        return reg


class Decompose(nn.Module):
    """
    A full decomposition model composed of multiple stacked DecompBlock modules.

    Each DecompBlock progressively removes seasonal and trend components from the time series.
    The goal is to obtain a fully denoised version of the input.

    Args:
    - depth (int): Number of DecompBlock layers to stack.
    - trend_kernel_size (int): Kernel size for Trend extraction in each block.
    - seq_len (int): Length of the input time series.
    - seasons (int): Number of separate Seasonal modules in each block.

    Input:
    - x (torch.Tensor): Input tensor of shape [batch_size, channels, seq_len].

    Output:
    - x (torch.Tensor): Final denoised time series after all DecompBlock layers.
    """

    def __init__(self, depth, trend_kernel_size, seq_len, seasons):
        super(Decompose, self).__init__()

        # Stack multiple DecompBlock modules
        self.blocks = nn.ModuleList([
            DecompBlock(trend_kernel_size, seq_len, seasons) for _ in range(depth)
        ])

    def forward(self, x):
        # Sequentially apply each DecompBlock
        for block in self.blocks:
            x = block(x)
        return x

    def symmetry_regularizer(self):
        """
        Compute the total symmetry regularization term for all blocks.

        Returns:
        - reg (torch.Tensor): Scalar tensor representing the total symmetry regularization.
        """
        reg = 0
        for block in self.blocks:
            reg += block.symmetry_regularizer()
        return reg


class Model(nn.Module):
    """
    Full time series forecasting model combining denoising decomposition and a prediction head.

    The model consists of:
    - A denoising encoder (stack of Decompose blocks), which removes seasonal and trend components.
    - A prediction head (either Linear or LowRank layer) that maps the cleaned sequence to the forecast horizon.

    The model supports:
    - Individual mode: separate decomposition and prediction heads per input channel.
    - Shared mode: single decomposition and prediction head for all channels.

    Args:
    - configs: Configuration object with the following attributes:
        - seq_len (int): Input sequence length.
        - pred_len (int): Prediction horizon.
        - enc_in (int): Number of input channels (features).
        - individual (bool): Whether to use individual models per channel.
        - decomposer_depth (int): Number of DecompBlock layers.
        - kernel_size (int): Kernel size for Trend extraction.
        - seasons (int): Number of Seasonal modules per DecompBlock.
        - rank (int): Rank for LowRank layer.
        - enable_lowrank (bool): Whether to use LowRank instead of Linear layer.
        - bias (bool): Whether to include bias in Linear or LowRank layer.

    Input:
    - x (torch.Tensor): Input tensor of shape [batch_size, seq_len, channels].

    Output:
    - out (torch.Tensor): Forecasted tensor of shape [batch_size, pred_len, channels].
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # Save configuration parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = configs.individual

        decomposer_depth = configs.decomposer_depth
        trend_kernel_size = configs.kernel_size
        seasons = configs.seasons
        rank = configs.rank
        enable_low_rank = configs.enable_lowrank

        
        # Create Decomposer Block per channel
        self.denoising_encoder = nn.ModuleList([
            Decompose(decomposer_depth, trend_kernel_size, self.seq_len, seasons)
            for _ in range(self.channels)
        ])

        if self.individual:
            if enable_low_rank:
                self.pred = nn.ModuleList([
                    LowRank(
                        in_features=self.seq_len,
                        out_features=self.pred_len,
                        rank=rank,
                        bias=configs.bias
                    )
                    for _ in range(self.channels)
                ])
            else:
                self.pred = nn.ModuleList([
                    nn.Linear(
                        in_features=self.seq_len,
                        out_features=self.pred_len,
                        bias=configs.bias
                    )
                    for _ in range(self.channels)
                ])
        else:
            if enable_low_rank:
                self.pred = LowRank(
                    in_features=self.seq_len,
                    out_features=self.pred_len,
                    rank=rank,
                    bias=configs.bias
                )
            else:
                self.pred = nn.Linear(
                    in_features=self.seq_len,
                    out_features=self.pred_len,
                    bias=configs.bias
                )

    def forward(self, x):
        # Permute to shape [batch_size, channels, seq_len]
        x = x.permute(0, 2, 1)

        # Compute and subtract mean for normalization (zero-mean input)
        seq_mean = torch.mean(x, axis=-1, keepdim=True)
        x = x - seq_mean

        # Denoising step: apply decomposition per channel
        clean_channels = []
        for i in range(self.channels):
            xi = x[:, i, :].contiguous() # [B, L]
            xi_clean = self.denoising_encoder[i](xi.unsqueeze(1))
            clean_channels.append(xi_clean)       # [B, 1, L]
        x_clean = torch.cat(clean_channels, dim=1) # [B, C, L]
       
        # Prediction step
        if self.individual:
            pred_channels = []
            for i in range(self.channels):
                pred_i = self.pred[i](x_clean[:, i, :])  # [B, H]
                pred_channels.append(pred_i.unsqueeze(1))         # [B, 1, H]
            out = torch.cat(pred_channels, dim=1)                 # [B, C, H]
        else:
            # Shared prediction head
            out = self.pred(x_clean)               # [B, C, H]

        # Restore mean
        out = out + seq_mean

        return out.permute(0,2,1)  # [B, H, C]

    def symmetry_regularizer(self):
        """
        Compute total symmetry regularization term for the model.

        Returns:
        - reg (torch.Tensor): Scalar tensor representing the total symmetry regularization.
        """
        reg = 0.0
        
        # Sum symmetry regularization from each channel-specific encoder
        for i in range(self.channels):
            reg += self.denoising_encoder[i].symmetry_regularizer()
        
        return reg
