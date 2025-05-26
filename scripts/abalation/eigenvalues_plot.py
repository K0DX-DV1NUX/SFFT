import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import pandas as pd


checkpoint = torch.load("/Users/adityadey/Documents/MyProjects/checkpoint.pth", map_location='cpu')

print(checkpoint.keys())

lambdas = [
    checkpoint[f'denoising_encoder.blocks.0.seasonal_list.{i}.diagonal'].resolve_conj().numpy()
    for i in range(3)
]

# Load F, F_inv, and lambdas from the checkpoint and resolve conjugate
f_modes = [
    checkpoint[f'denoising_encoder.blocks.0.seasonal_list.{i}.f_modes']#.resolve_conj().numpy()
    for i in range(3)
]
f_modes_inv = [
    checkpoint[f'denoising_encoder.blocks.0.seasonal_list.{i}.f_modes_inv']#.resolve_conj().numpy()
    for i in range(3)
]

print(torch.allclose((f_modes[0] @ f_modes_inv[0]).real, torch.eye(f_modes[0].shape[0]), atol=1e-2))


# X = np.stack(lambdas, axis=1)  # Shape: (L, 3)


## Reconstruct circulant matrices and extract first row
# first_rows = []
# for i in range(3):
#     Lambda = np.diag(lambdas[i].astype(np.complex64))
#     F_inv = f_modes_inv[i].astype(np.complex64)
#     F = f_modes[i].astype(np.complex64)
#     C = F_inv @ Lambda @ F
#     first_row = C.real[0]  # Extract the real part of the first row
#     first_rows.append(first_row)




# def moving_average(x, w=5):
#     return np.convolve(x, np.ones(w)/w, mode='same')

# # Smooth the first rows
# smoothed_rows = [moving_average(row, w=10) for row in first_rows]

# # Plot the smoothed rows
# plt.figure(figsize=(8, 4))
# colors = ['blue', 'magenta', 'green']
# labels = ['Seasonal 1', 'Seasonal 2', 'Seasonal 3']
# for i, row in enumerate(smoothed_rows):
#     plt.plot(row, label=labels[i], color=colors[i], linewidth=2)

# plt.xlabel('Lag Index')
# plt.ylabel('Weight')
# plt.title('Smoothed First Row of Circulant Matrices (Impulse Response)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# #plt.savefig("circulant_first_rows_smoothed.png", dpi=300)
# plt.show()

for i, lam in enumerate(lambdas):
    energy = np.sum(np.abs(lam)**2)
    print(f"Energy of Seasonal {i+1}: {energy:.4f}")

# C1 = f_modes_inv[0] @ np.diag(lambdas[0]) @ f_modes[0]
# C2 = f_modes_inv[1] @ np.diag(lambdas[1]) @ f_modes[1]
# C3 = f_modes_inv[2] @ np.diag(lambdas[2]) @ f_modes[2]

# C = [C1, C2, C3]

# for i, c in enumerate(C):
#     full_c = c.real.flatten()
#     energy = np.sum(np.abs(full_c)**2)
#     print(f"Energy of Full Circulant Matrix {i+1}: {energy:.4f}")

# print(np.allclose(f_modes[0] @ f_modes_inv[0], torch.eye(f_modes[0].shape[0]), atol=1e-2))

# def _fourier_basis(N):
       
#         """Generate the normalized DFT matrix F of size (N, N).
#         F_{jk} = 1/sqrt(N) * exp(-2 * pi * i * j * k / N)
#         """
#         # Generate the row indices (j) and column indices (k)
#         j = torch.arange(N).view(-1, 1)  # shape (N, 1)
#         k = torch.arange(N).view(1, -1)  # shape (1, N)

#         # Compute the complex exponential part (exp(-2 * pi * i * j * k / N))
#         exponent = -2j * torch.pi * j * k / N
#         return (1 / torch.sqrt(torch.tensor(N, dtype=torch.float32))) * torch.exp(exponent)

# fourier1 = _fourier_basis(336)
# fourier1_T = torch.conj(fourier1).T

# print(torch.allclose(fourier1 @ fourier1_T, torch.eye(fourier1.shape[0]).to(torch.complex64), atol=1e-4))


# f_modes[0]