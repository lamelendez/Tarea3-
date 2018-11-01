import numpy as np
import matplotlib.pyplot as plt


from matplotlib.colors import LogNorm

foto = plt.imread("Arboles.png") #Descargo la imagen en un array 2d 

f = np.fft.fft2(foto)
fshift = np.fft.fftshift(f)
freq = np.fft.fftfreq(len(f[0]))
magnitude_spectrum = (30*np.abs(fshift))
plt.imshow(magnitude_spectrum, norm=LogNorm(20))
plt.savefig("MelendezLaura_FT2D.pdf")
plt.show()

#plt.figure()
#plt.plot(freq,abs(fshift))


def filtro(fu,fe):
    for i in range(np.shape(fe)[0]):
        for j in range(np.shape(fe)[0]): 
            if(abs(fu[i,j])>2000.0 and abs(fu[i,j])<5000.0):
                fu[i,j]=0.0
    return fu

plt.figure()
plt.plot(freq,filtro(fshift,freq))
plt.savefig("LauraMelendez_FT2D_filtrada.pdf")
#plt.show()


#ahora vamos a hacer la inversa filtrada

fa = np.fft.ifft2(foto)
fshifta = np.fft.fftshift(fa)
magnitude_spectrum = (30*np.abs(fshifta))
plt.imshow(magnitude_spectrum, norm=LogNorm(20)) 
plt.show()
plt.savefig("MelendezLaura_Imagen_filtrada.pdf")


