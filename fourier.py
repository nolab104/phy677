import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftshift, ifft, ifftshift

# sampling period -- change this to visualize the effects of aliasing. 
T = 0.2
# number of samples ... keep this even
N = 320
# frequency step
delta_nu = 1.0 / (N * T)

# locations where the function is sampled 
t = np.arange(-N / 2, N / 2) * T
# frequency sampling locations
nu = np.arange(-N / 2, N / 2) * delta_nu

a = 0.5 # scaling the Gaussian
f = np.exp(-a * t**2.0) #function 
F = np.sqrt(np.pi / a) * np.exp(-np.pi**2.0 * nu**2.0 / a) # FT of the function

F_fft = fft(f) # fft, no shifting at this juncture. 

f_ifft = ifft(F_fft)
fig, axs = plt.subplots(figsize = (8,4), ncols = 2, nrows = 1)
# plot the function and its samples
axs[0].plot(t,f,"-")
axs[0].plot(t,f_ifft,"--o")
axs[0].set_xlabel("t")
axs[0].set_title("The function")
# Fourier space plot
axs[1].plot(nu,F,"-")
axs[1].plot(nu,T*fftshift(F_fft),"s")
axs[1].set_xlabel(r"$\nu$")
axs[1].set_ylim(bottom = 0)
axs[1].set_title("The Fourier Transform")
plt.show()