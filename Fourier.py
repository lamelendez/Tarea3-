import numpy as np 
import matplotlib.pyplot as plt
import plotly

from scipy.interpolate import interp1d
from scipy.fftpack import fft, fftfreq

#guardo los archivos
datos1 = np.genfromtxt("signal.dat",delimiter=",")
datos2 = np.genfromtxt("incompletos.dat", delimiter=",")

#grafica sin modificar de signal.dat
plt.figure()
plt.plot(datos1[:,0],datos1[:,1])
plt.title("datos originales")
plt.savefig("MelendezLaura_signal.pdf")

#mi funcion propia de fourier discreta modificada para un array y no una funcion

def fourierdiscreta(f):
    funcion = 0
    N = np.shape(f)[0]
    n = np.arange(N)
    f1 = f[:,1]
    f2 = f[:,0]            
    funcion = (np.exp((-2j*np.pi*n*f2)/N))
    funcionr = f1*funcion
    return funcionr


    
print(fourierdiscreta(datos1))


#vamos a volver reales los valores del array que salen por fourier discreta par apoder graficarlos

def reales(datox):
    f2 = np.array([]) 
    for i in range(np.shape(datox)[0]):
         f = (datox[i].real**2) + (datox[i].imag**2)
         f1 = np.sqrt(f)
         f2 = np.append(f2,f1)
    return f2 


#n = np.linspace(1,513,512)    
#print(reales(fourierdiscreta(datos1,512,k)))




#con fft sin bonito 

n = 512 #numero de datos que tenemos 
dt = (datos1[:,1][2] - datos1[:,1][1]) #numero de datos por unidad de frecuenca
fft_x = fft(datos1[:,1]) / n # FFT Normalized
freq = fftfreq(n, dt) # Recuperamos las frecuencias
plt.figure()
plt.plot(freq,abs(fft_x))
plt.savefig("LauraMelendez_TF.pdf") 

#plt.figure()
#plt.plot(freq,reales(fourierdiscreta(datos1)))              #Esta es mi señal discreta pero no funciona entonces no la grafiqué

#plt.savefig("MelendezLaura_TF.pdf")
#plt.show()
#print(fourierdiscreta(datos1,512,k))



#las hallé con la grafica que me dio, los picos.
print("las frecuencias principales de mi señal son: 0.014452 y 0.038561 ")


#Filtro pasabajas


clean_f = np.fft.ifft(datos1[:,1])

def pasabajas(fe,fu,fcc):       
    for i in range(np.shape(fe)[0]):
        if fu[i]>fcc:   
            fu[i]=0.0
    return fu   


#con la inversa de fft 

plt.figure() 
pb = pasabajas(freq,clean_f,100)
plt.plot(pb)
plt.savefig("MelendezLaura_filtrada.pdf")


print("No se puede hacer la transformada con esos datos porque las potencias con las que trabaja fourier deben ser de dos. Estos datos tienen una longitud de 117. LA interpolacion es necesaria") 


#Interpolacion de mis datos incorrectos 

def interpolar (x,y):
    xnuevo = np.linspace(0.000390625,0.028515625,512) #primero y final de la columan 0 de los archivos incompletos

    #flin = interp1d(x, y)
    fcua = interp1d(x, y,kind ='quadratic')
    fcub = interp1d(x, y,kind ='cubic')
    
    return xnuevo, fcua , fcub

x,y1,y2 = interpolar(datos2[:,0],datos2[:,1])

cuadratica = fft(y1(x))/len(x)
cubico = fft(y2(x))/len(x)          #fourier de las interpolaciones

  #ploteo de eso
plt.figure()
plt.subplot(221)
plt.plot(freq,abs(fft_x))
plt.title("señal")
plt.subplot(222)
plt.plot(freq,abs(cuadratica))
plt.title("señal interpolada cuadratica")
plt.subplot(223)
plt.plot(freq,abs(cubico))
plt.title("señal interpolada cubica")
plt.savefig("MelendezLaura_TF_interpola.pdf")
   


 #ahora quitandole el ruido 

plt.subplot(321)
pb1= pasabajas(freq,abs(cuadratica),500)
plt.plot(pb1)
plt.title("cuadratica pasabajas 500hz")
plt.subplot(322)
pb2= pasabajas(freq,abs(cubico),500)
plt.plot(pb2)
plt.title("cubico pasabajas 500hz")
plt.subplot(323)
pb3= pasabajas(freq,fft_x,500)
plt.plot(pb3)
plt.title("señal pasabajas 500hz")
plt.subplot(324)
pb4= pasabajas(freq,fft_x,1000)
plt.plot(pb4)
plt.title("señal pasabajas 1000hz")
plt.subplot(325)
pb5= pasabajas(freq,abs(cubico),1000)
plt.plot(pb5)
plt.title("cubico pasabajas 1000hz")
plt.subplot(326)
pb6= pasabajas(freq,abs(cuadratica),1000)
plt.plot(pb6)
plt.title("cuadratica pasabajas 1000hz")
plt.savefig("MelendezLaura_2Filtros.pdf")









