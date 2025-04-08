import torch
import torch.nn as nn
import torch.nn.functional as F



class Trend(nn.Module):

    def __init__(self, kernel_size, stride):
        super(Trend, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    # x: [batch_size, channels, seq_len]
    def forward(self, x):
        #padding on front end of time series
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1))

        # padding on the both ends of time series
        #front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        #end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        #x = torch.cat([front, x, end], dim=1)

        x = torch.cat([front, x], dim=-1)
        x = self.avg(x)
        return x # [batch_size, channels, seq_len]
    
    
class Seasonal(nn.Module):

    def __init__(self, seq_len, f_basis):
        super(Seasonal, self).__init__()
        self.seq_len = seq_len
        self.register_buffer('f_modes', f_basis)
        self.register_buffer('f_modes_inv', torch.conj(self.f_modes).T)
        
        # Initialize the diagonal to enforce symmetry
        self.diagonal = nn.Parameter(nn.init.uniform_(torch.empty(seq_len), a=-1.0, b=1.0), requires_grad=True)

    
    def forward(self, x):

        # Now diagonal_full is symmetric: c_j = c_{n-j}
        diagonal_mat = torch.diag(self.diagonal).to(torch.complex64)

        circulant_mat = self.f_modes @ diagonal_mat @ self.f_modes_inv
        x = x @ circulant_mat.real
        return F.tanh(x)

    @staticmethod
    def _fourier_basis(N):
       
        """Generate the normalized DFT matrix F of size (N, N).
        F_{jk} = 1/sqrt(N) * exp(-2 * pi * i * j * k / N)
        """
        # Generate the row indices (j) and column indices (k)
        j = torch.arange(N).view(-1, 1)  # shape (N, 1)
        k = torch.arange(N).view(1, -1)  # shape (1, N)

        # Compute the complex exponential part (exp(-2 * pi * i * j * k / N))
        exponent = -2j * torch.pi * j * k / N
        return (1 / torch.sqrt(torch.tensor(N, dtype=torch.float32))) * torch.exp(exponent)
    
    def symmetry_regularizer(self):
        diagonal_mat = torch.diag(self.diagonal).to(torch.complex64)
        circulant_mat = self.f_modes @ diagonal_mat @ self.f_modes_inv

        # Calculate the symmetry regularization term
        circulant_mat = circulant_mat.real
        return (circulant_mat - circulant_mat.T).norm(p=2) # Frobenius norm of the difference


class EncoderBlock(nn.Module):
    
    def __init__(self, trend_kernel_size, seq_len, seasons):
        super(EncoderBlock, self).__init__()
        self.trend = Trend(kernel_size=trend_kernel_size, stride=1)
        
        f_modes = Seasonal._fourier_basis(seq_len*seasons)
        seasonal_f_modes = [f_modes[i*seq_len:(i+1)*seq_len, i*seq_len:(i+1)*seq_len] for i in range(seasons)]
        
        self.seasonal_list = nn.ModuleList([
            Seasonal(seq_len, seasonal_f_modes[_]) for _ in range(seasons)
        ])

    # x: [batch_size, channels, seq_len]
    def forward(self, x):
        # Apply trend extraction
        x_T = self.trend(x)

        x_rem = x - x_T

        x_S = []
        for i in range(len(self.seasonal_list)):
            x_seasonal = self.seasonal_list[i](x_rem)
            x_S.append(x_seasonal)
            x_rem = x_rem - x_seasonal
        
        # De-noised Time series
        return x - x_rem
    
    def symmetry_regularizer(self):
        # Compute the symmetry regularization term for all seasonal components
        reg = 0
        for seasonal in self.seasonal_list:
            reg += seasonal.symmetry_regularizer()
        return reg

class DenoisingEncoder(nn.Module):

    def __init__(self, depth, trend_kernel_size, seq_len, seasons):
        super(DenoisingEncoder, self).__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(trend_kernel_size, seq_len, seasons) for _ in range(depth)
        ])

    # x: [batch_size, channels, seq_len]
    def forward(self, x):

        
        for block in self.blocks:
            x = block(x)

        return x
    
    def symmetry_regularizer(self):
        # Compute the symmetry regularization term for all blocks
        reg = 0
        for block in self.blocks:
            reg += block.symmetry_regularizer()
        return reg



class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len  # Input sequence length
        self.pred_len = configs.pred_len  # Prediction horizon
        self.channels = configs.enc_in  # Number of input channels (features)
        
        encoder_depth = 3
        trend_kernel_size = 25
        seasons = 3
        self.denoising_encoder = DenoisingEncoder(encoder_depth, trend_kernel_size, self.seq_len, seasons)
        self.pred = nn.Linear(self.seq_len, self.pred_len)

    # x: [batch_size, seq_len, channels] 
    def forward(self, x):
        
        x = x.permute(0, 2, 1) # [batch_size, channels, seq_len]
        x = self.denoising_encoder(x)
        out = self.pred(x) # [batch_size, channels, pred_len]
        out = out.permute(0, 2, 1)
        return out

    def symmetry_regularizer(self):
        # Compute the symmetry regularization term for the entire model
        reg = self.denoising_encoder.symmetry_regularizer()
        return reg