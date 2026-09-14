# Demonstration of sampling in 2D, and propagation of scalar field through a 4f imaging system. 

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift, ifft, ifftshift, fft2, ifft2

#
lambda0 = 0.5
NA = 1.0
# sampling period
dx = 0.1
dy = 0.1
# number of samples ... keep this even
N = 512
#  spatial frequency step (calculated)
dkx = 1.0 / (N * dx)
dky = 1.0 / (N * dy)
# spatial grid (calculated)
x_values = np.arange(-N / 2, N / 2) * dx
y_values = np.arange(-N / 2, N / 2) * dy
# spatial frequency grid (calculated)
kx_values = np.arange(-N / 2, N / 2) * dkx
ky_values = np.arange(-N / 2, N / 2) * dky
# This is a grid of xy and kx-ky points
x, y = np.meshgrid(x_values, y_values, indexing="ij")
kx, ky = np.meshgrid(kx_values, ky_values, indexing="ij")

# rectangular object
E0 = np.double(np.abs(x) < 8) * np.double(np.abs(y) < 4)
# its Fourier transform
F0 = fft2(E0)

# apply pupil as a kappa_perp limit
P = np.double(kx**2.0 + ky**2.0 < (NA / lambda0) ** 2.0)
# the Fourier plane field
E1 = F0 * fftshift(P)
# angular spectrum beyond the Fourier plane
F1 = fft2(E1)
# image field
E2 = F1

# Plots
fig, axs = plt.subplots(figsize=(8, 4), ncols=2, nrows=2)
axs[0, 0].imshow(
    np.double(E0),
    extent=[
        dx * (-N / 2 - 0.5),
        dx * (N / 2 + 0.5),
        dy * (-N / 2 - 0.5),
        dy * (N / 2 + 0.5),
    ],
)
axs[0, 1].imshow(
    np.abs(fftshift(np.double(E1))),
    extent=[
        dkx * (-N / 2 - 0.5),
        dkx * (N / 2 + 0.5),
        dky * (-N / 2 - 0.5),
        dky * (N / 2 + 0.5),
    ],
)
axs[1, 0].imshow(
    P,
    extent=[
        dkx * (-N / 2 - 0.5),
        dkx * (N / 2 + 0.5),
        dky * (-N / 2 - 0.5),
        dky * (N / 2 + 0.5),
    ],
)
kappa_perp_limit = 1.0 / lambda0
th = np.linspace(0, 360, 361) * np.pi / 180
cx = np.cos(th) * kappa_perp_limit
sx = np.sin(th) * kappa_perp_limit
axs[1, 0].plot(cx, sx)

axs[1, 1].imshow(
    (np.abs(np.double(E2) ** 2.0)),
    extent=[
        dx * (-N / 2 - 0.5),
        dx * (N / 2 + 0.5),
        dy * (-N / 2 - 0.5),
        dy * (N / 2 + 0.5),
    ],
    cmap="gray",
)
# axs[1, 1].set_xlim(-2, 2)
# axs[1, 1].set_ylim(-2, 2)
plt.show()
