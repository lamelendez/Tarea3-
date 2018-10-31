import numpy as np 
import matplotlib.pyplot as plt 
import plotly.plotly as py
import plotly.tools as tls
from scipy.interpolate import interp1d
from scipy.fftpack import fft, fftfreq

#guardo los archivos
datos1 = np.genfromtxt("signal.dat",delimiter=",")
datos2 = np.genfromtxt("incompletos.dat", delimiter=",")

#grafica sin modificar de signal.dat
plt.figure()
plt.plot(datos1[:,0],datos1[:,1])
plt.savefig("MelendezLaura_signal.pdf")

#mi funcion de fourier discreta modificada para un array y no una funcion

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

#la transformada de Fourier de los datos de la seÑal usando mi implementacion (grafica sin mostrar)

plt.figure()
plt.plot(datos1[:,0],reales(fourierdiscreta(datos1)))
#plt.show()
#print(fourierdiscreta(datos1,512,k))
plt.savefig("MelendezLaura_TF.pdf")

#con fft sin bonito 

n = 512 #numero de datos que tenemos 
dt = 2*np.pi/(0.0272321428571-0.0271763392857) #numero de datos por unidad de frecuenca
fft_x = fft(datos1[:,1]) / n # FFT Normalized
freq = fftfreq(n, dt) # Recuperamos las frecuencias
plt.figure()
plt.plot(freq,abs(fft_x))
plt.show()

#las hallé con la grafica que me dio, los picos.
print("las frecuencias principales de mi señal son: -1.855e-07,-6.7649e-08,8.96178e-08,2.07567e-07 ")


#Filtro pasabajas

freq_cut = 1000
fft_x[abs(freq) > freq_cut] = 0

#con la inversa de fft 

plt.figure()
t = np.linspace(n-512,n,n) 
clean_f = np.fft.ifft(fft_x) 

plt.plot(t,np.real(clean_f), linewidth=5, color='green')
plt.show()


#LAURA ACUERDESE QUE LE FALTA EXPLICAR POR QUÉ LOS DATOS ESOS DE MERGA NO DAN 


#Interpolacion de mis datos incorrectos 

def interpolar (x,y):
    xnuevo = np.linspace(0.000390625,0.028515625,512)
    #flin = interp1d(x, y)
    fcua = interp1d(x, y,kind ='quadratic')
    fcub = interp1d(x, y,kind ='cubic')
    
    return xnuevo, fcua , fcub

x,y1,y2 = interpolar(datos2[:,0],datos2[:,1])

na = 512 #numero de datos que tenemos 
dta = 2*np.pi/(0.0205915178571-0.0205357142857) #numero de datos por unidad de frecuenca
fft_xa = fft(y1) / na # FFT Normalized
freqa = fftfreq(na, dta) # Recuperamos las frecuencias
plt.figure()





fft_xb = fft(y2) / n # FFT Normalized
freqb = fftfreq(na, dta) # Recuperamos las frecuencias
  

fig = plt.figure()

ax1 = fig.add_subplot(221)
ax1.plot(freqa,abs(fft_xa))

ax2 = fig.add_subplot(222)
ax2.plot(freqb,abs(fft_xb))

ax3 = fig.add_subplot(223)
ax3.plot(datos1[:,0],reales(fourierdiscreta(datos1)))

plt.tight_layout()
fig = plt.gcf()

plotly_fig = tls.mpl_to_plotly( fig )
plotly_fig['layout']['title'] = 'Transformadas de fourier '
plotly_fig['layout']['margin'].update({'t':40})

py.iplot(plotly_fig)
plt.show()   








