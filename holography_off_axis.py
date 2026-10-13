import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift, ifft, ifftshift, fft2, ifft2

plt.rcParams["font.family"] = "Arial"

# all units are micrometres

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
x, y = np.meshgrid(x_values, y_values, indexing="xy")
kx, ky = np.meshgrid(kx_values, ky_values, indexing="xy")

kappa = 1.0/lambda0

# elliptical transparent object with a polynomial phase
u = x / 10.0
v = y / 6.0
phi = (u**3 + 4 * u * v**2.0 - 4 * u**2 * v + v**3.0) # some phase function 
E0 = np.exp(1j * phi * np.double(u**2.0 + v**2.0 < 1)) # the field at the object plane
nq = 51 # carrier in units of dkx, so that it is periodic on the grid
q = nq * dkx # our reference wavevector is an integer multiple of the spatial sampling frequency
Er = np.exp(1j * 2 * np.pi * q * x)
I0 = np.abs(E0 + Er)**2.0
I0hat = fftshift(fft2(ifftshift(I0))) # intensity spectrum, origin at the centre

# pick up the sideband with a circular window
kcut = 0.5
H = np.double((kx + q) ** 2.0 + ky ** 2.0 < kcut ** 2.0)
# shift it to the centre in k-space, and transform back
E1hat = np.roll(I0hat * H, nq, axis=1)
E1 = fftshift(ifft2(ifftshift(E1hat)))

# repeat for the other sideband
H2 = np.double((kx - q) ** 2.0 + ky ** 2.0 < kcut ** 2.0)
E2hat = np.roll(I0hat * H2, -nq, axis=1)
E2 = fftshift(ifft2(ifftshift(E2hat)))

# Plots
fig, axs = plt.subplots(figsize=(12, 6), ncols=3, nrows=2)
axs[0, 0].imshow(
    np.angle(E0),
    extent=[
        dx * (-N / 2 - 0.5),
        dx * (N / 2 + 0.5),
        dy * (-N / 2 - 0.5),
        dy * (N / 2 + 0.5),
    ],
    origin="lower",
)
axs[0, 1].imshow(
    I0,
    extent=[
        dx * (-N / 2 - 0.5),
        dx * (N / 2 + 0.5),
        dy * (-N / 2 - 0.5),
        dy * (N / 2 + 0.5),
    ],
    origin="lower",
)
axs[0, 2].imshow(
    np.log(1 + np.abs(I0hat)),
    extent=[
        dkx * (-N / 2 - 0.5),
        dkx * (N / 2 + 0.5),
        dky * (-N / 2 - 0.5),
        dky * (N / 2 + 0.5),
    ],
    origin="lower",
)
th = np.linspace(0, 360, 361) * np.pi / 180
cx = np.cos(th) * kcut - q
sx = np.sin(th) * kcut
axs[0, 2].plot(cx, sx, "r")
axs[0, 2].plot(cx + 2 * q, sx, "c")
axs[1, 0].imshow(
    np.log(1 + np.abs(E1hat)),
    extent=[
        dkx * (-N / 2 - 0.5),
        dkx * (N / 2 + 0.5),
        dky * (-N / 2 - 0.5),
        dky * (N / 2 + 0.5),
    ],
    origin="lower",
)
axs[1, 1].imshow(
    np.angle(E1),
    extent=[
        dx * (-N / 2 - 0.5),
        dx * (N / 2 + 0.5),
        dy * (-N / 2 - 0.5),
        dy * (N / 2 + 0.5),
    ],
    origin="lower",
)
axs[1, 2].imshow(
    np.angle(E2),
    extent=[
        dx * (-N / 2 - 0.5),
        dx * (N / 2 + 0.5),
        dy * (-N / 2 - 0.5),
        dy * (N / 2 + 0.5),
    ],
    origin="lower",
)

# titles
axs[0, 0].set_title("The phase of the transparent object")
axs[0, 1].set_title("The interference pattern")
axs[0, 2].set_title("The Intensity Spectrum")
axs[1, 0].set_title("The sideband moved to the center")
axs[1, 1].set_title("Reconstructed Object")
axs[1, 2].set_title("Reconstruction from the other sideband")

# colorbar for each image
for ax in axs.flat:
    if ax.images:
        fig.colorbar(ax.images[0], ax=ax)
plt.tight_layout()
plt.show()
