import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

foto = plt.imread("Arboles.png") #Descargo la imagen en un array 2d 

f = np.fft.fft2(foto)
fshift = np.fft.fftshift(f)
freq = np.fft.fftfreq(len(f[0]))
magnitude_spectrum = (25*np.abs(fshift))
plt.imshow(magnitude_spectrum, norm=LogNorm(20))
plt.savefig("MelendezLaura_FT2D.pdf")
plt.show()

plt.figure()
plt.plot(freq,abs(fshift))
plt.show()

def filtro(
    for i in range(np.shape(freq)[0]):
        if(
