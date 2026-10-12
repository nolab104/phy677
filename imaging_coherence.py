import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift, ifft, ifftshift, fft2, ifft2

#
lambda0 = 0.5
NA = 0.5
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
E0 = np.double(np.abs(x+2*y) < 8) * np.double(np.abs(x-3*y) < 4)
E0 = np.double(np.abs(x) < 8) * np.double(np.abs(y) < 4)
I0 = np.abs(E0)**2.0

# its Fourier transform
F0i = fft2(I0) # intensity spectrum

# apply pupil as a kappa_perp limit
H = np.double(kx**2.0 + ky**2.0 < (NA / lambda0) ** 2.0)
OTF = ifft2(np.abs(fft2(H))**2.0)

# the Fourier plane field
F1i = F0i * (OTF)

# field beyond the Fourier plane
I1 = fft2(F1i)

# image field
I2 = fft2(F1i)

# Plots
fig, axs = plt.subplots(figsize=(8, 4), ncols=2, nrows=2)
axs[0, 0].imshow(
    np.double(I0),
    extent=[
        dx * (-N / 2 - 0.5),
        dx * (N / 2 + 0.5),
        dy * (-N / 2 - 0.5),
        dy * (N / 2 + 0.5),
    ],
)
axs[0,1].imshow(
    ifftshift(np.abs(OTF)),
    extent=[
        dkx * (-N / 2 - 0.5),
        dkx * (N / 2 + 0.5),
        dky * (-N / 2 - 0.5),
        dky * (N / 2 + 0.5),
    ],
    cmap="gray",
)
kappa_perp_limit = 1.0 / lambda0
th = np.linspace(0, 360, 361) * np.pi / 180
cx = np.cos(th) * kappa_perp_limit
sx = np.sin(th) * kappa_perp_limit
axs[1,0].plot(cx, sx)


axs[1,0].imshow(
    np.abs(F1i),
    extent=[
        dkx * (-N / 2 - 0.5),
        dkx * (N / 2 + 0.5),
        dky * (-N / 2 - 0.5),
        dky * (N / 2 + 0.5),
    ],
    cmap="gray",
)
kappa_perp_limit = 1.0 / lambda0
th = np.linspace(0, 360, 361) * np.pi / 180
cx = np.cos(th) * kappa_perp_limit
sx = np.sin(th) * kappa_perp_limit
axs[1,0].plot(cx, sx)

axs[1,1].imshow(
    np.abs(fft2(fftshift(F1i))),
    extent=[
        dx * (-N / 2 - 0.5),
        dx * (N / 2 + 0.5),
        dy * (-N / 2 - 0.5),
        dy * (N / 2 + 0.5),
    ],
    cmap="gray"
)

# axs[1, 1].set_xlim(-2, 2)
# axs[1, 1].set_ylim(-2, 2)
plt.show()
