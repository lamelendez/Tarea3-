import numpy as np 
import matplotlib.pyplot as plt 

#guardo los archivos
datos1 = np.genfromtxt("signal.dat",delimiter=",")
datos2 = np.genfromtxt("incompletos.dat", delimiter=",")

#grafica sin modificar de signal.dat
plt.figure()
plt.plot(datos1[:,0],datos1[:,1])
plt.savefig("MelendezLaura_signal.pdf")

#mi funcion de fourier discreta modificada para un array y no una funcion

def fourierdiscreta(f,n):
    funcion = 0
    N = np.shape(n)[0]
    f1 = f[:,1]
    f2 = f[:,0]            
    funcion = f1*(np.exp((-2j*np.pi*n*f2)/N))
    return funcion

#vamos a volver reales los valores del array que salen por fourier discreta par apoder graficarlos

def reales(datox):
    f2 = np.array([]) 
    for i in range(np.shape(datox)[0]):
         f = (datox[i].real**2) + (datox[i].imag**2)
         f1 = np.sqrt(f)
         f2 = np.append(f2,f1)
    return f2 


n = np.linspace(1,513,512)    
#print(reales(fourierdiscreta(datos1,512,k)))

#la transformada de Fourier de los datos de la seÑal usando mi implementacion (grafica sin mostrar)

plt.figure()
plt.plot(datos1[:,0],reales(fourierdiscreta(datos1,n)))
#plt.show()
#print(fourierdiscreta(datos1,512,k))
plt.savefig("MelendezLaura_TF.pdf")

#con fft 



