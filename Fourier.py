import numpy as np 
import matplotlib.pyplot as plt 

#guardo los archivos
datos1 = np.genfromtxt("signal.dat",delimiter=",")
datos2 = np.genfromtxt("incompletos.dat", delimiter=",")

#grafica sin modificar de signal.dat
plt.plot(datos1[:,0],datos1[:,1])
plt.savefig("MelendezLaura_signal.pdf")

#mi funcion de fourier discreta modificada para un array y no una funcion

def fourierdiscreta(f,n,k):
    funcion = 0
    f1 = f[:,1]
    f2 = f[:,0]
    for i in range(np.shape(f)[0]):         
        funcion += f1[i]*(np.exp((-2j*np.pi*k*f2[i])/n))
    return funcion

k = np.linspace(0,2*np.pi,512)
plt.plot(k,fourierdiscreta(datos1,512,k))
plt.show()
#print(fourierdiscreta(datos1,512,k))


