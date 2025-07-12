import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Grid and field setup
N = 64
x, y, z = np.indices((N, N, N))
center = np.array([N // 2, N // 2, N // 2])

# Create Gaussian Random Field
np.random.seed(0)
G = np.random.normal(0, 3, (N, N, N))

# --- Plot a 2D slice of the Gaussian Random Field ---
plt.figure(figsize=(5, 4))
plt.imshow(G[N // 2], cmap='viridis')
plt.colorbar()
plt.title("Slice of Gaussian Random Field")
plt.tight_layout()
plt.show()

# --- 3D Scatter plot of Gaussian Random Field (low alpha for visibility) ---
G_flat = G.flatten()
xg, yg, zg = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing='ij')
x_flat = xg.flatten()
y_flat = yg.flatten()
z_flat = zg.flatten()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x_flat, y_flat, z_flat, c=G_flat, alpha=0.02, cmap='viridis')
plt.title("3D Scatter of Gaussian Random Field")
plt.colorbar(scatter, shrink=0.5)
plt.tight_layout()
plt.show()

# --- Fourier Transform of GRF ---
G_ff = np.fft.fftn(G)
G_fft = np.fft.fftshift(G_ff)
GF = np.abs(G_fft)

# --- Plot a 2D slice of the Fourier transform of GRF ---
plt.figure(figsize=(5, 4))
plt.imshow(GF[N // 2], cmap='magma')
plt.colorbar()
plt.title("Fourier Transform (magnitude) of GRF")
plt.tight_layout()
plt.show()

# --- Initialize for smoothing ---
radii = np.arange(0, N // 2 + 1)
means, variances = [], []

# Loop over radii for sphere convolution
for r in radii:
    # Create spherical mask
    dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2)
    sphere = (dist <= r).astype(float)

    # Normalize kernel if non-zero
    if np.sum(sphere) > 0:
        sphere /= np.sum(sphere)

    # Plot a slice of the spherical mask
    if r in [1, 5, 10, 20, 32]:  # selected slices for illustration
        plt.figure(figsize=(5, 4))
        plt.imshow(sphere[N // 2], cmap='Greys')
        plt.colorbar()
        plt.title(f"Slice of Sphere Mask (r={r})")
        plt.tight_layout()
        plt.show()

    # Fourier transform of the spherical mask
    S_f = np.fft.fftn(sphere)
    S_fft = np.fft.fftshift(S_f)
    S_abs = np.abs(S_fft)

    if r in [1, 5, 10, 20, 32]:  # selected slices for illustration
        plt.figure(figsize=(5, 4))
        plt.imshow(S_abs[N // 2], cmap='plasma')
        plt.colorbar()
        plt.title(f"FT Magnitude of Sphere (r={r})")
        plt.tight_layout()
        plt.show()

    # Multiply in Fourier space and inverse transform
    smoothed_fft = GF * S_abs
    smoothed_ifft = np.fft.ifftn(np.fft.ifftshift(smoothed_fft))
    smoothed_field = np.abs(smoothed_ifft)

    # Store mean and variance
    means.append(np.mean(smoothed_field))
    variances.append(np.var(smoothed_field))

    # Show smoothed result for selected radii
    if r in [1, 5, 10, 20, 32]:
        plt.figure(figsize=(5, 4))
        plt.imshow(smoothed_field[N // 2], cmap='viridis')
        plt.colorbar()
        plt.title(f"Smoothed Field Slice (r={r})")
        plt.tight_layout()
        plt.show()

# --- Plot Mean and Variance vs Radius ---
plt.figure(figsize=(6, 4))
plt.plot(radii, variances, 'b-o', label="Variance")
plt.xlabel("Radius of Smoothing Sphere")
plt.ylabel("Variance")
plt.title("Variance vs Radius")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(radii, means, 'g-o', label="Mean")
plt.xlabel("Radius of Smoothing Sphere")
plt.ylabel("Mean")
plt.title("Mean vs Radius")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
