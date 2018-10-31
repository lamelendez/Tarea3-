import numpy as np
import matplotlib.pyplot as plt 

#SEGUNDO PUNTO TAREA 3 ---------------------------------------------------------


#extraer los datos

data = (np.genfromtxt('WDBC.dat', delimiter = ','))[:,1:]
data1 = np.delete(data,0,1)
mb = (np.genfromtxt('WDBC.dat', delimiter = ',',dtype=str))[:,1]
mb1 = np.array([0 if e == "M" else 1 for e in mb]) #cambiar los strings por numeros
data[:,0]=mb1

#print(data[:,1])

#matriz covarianza

def matrizcov(datax):

    for i in range(len(datax)):
        datax[i] = (datax[i]-np.mean(datax[i]))/np.std(datax[i]) #normalizada

    n_dim = np.shape(datax)[1]  
    m_dim = len(datax[0])  
    covarianza = np.ones([n_dim,n_dim])
    for i in range(n_dim):
        for j in range(n_dim):        
            meani = (np.mean(datax[:,i]))  
            meanj = (np.mean(datax[:,j]))	    	
            #varianza=np.var(datax[i])
            #d = np.sqrt(varianza)
            covarianza[i,j] = np.sum((datax[:,i]-meani)*(datax[:,j]-meanj))/(m_dim-1)
    return covarianza 
print(matrizcov(data1)[0])

A = matrizcov(data1)

val,vec = np.linalg.eig(A)

#print(val)

#calcular los autovectores de sus autovalores

#for i in range(np.shape(val)[0]):
 #   print("el vector", vec[:,i], "corresponde al valor" , val[i])
    

#hallaremos los parametros más importantes con los porcentajes            



def porcentajemayores(array):  
    total = np.sum(array)
    narray = np.array([])
    nvalores = np.array([])
    auto = np.array([])
    for i in range(np.shape(array)[0]):
        totali = (array[i] * 100)/total
        if(totali>=15): #tocó sobre el 15% para escoger dos 
             narray = np.append(narray,totali)             
             auto = np.append(auto,i+1)


    return "los porcentajes de los mayores componentes son:", narray , "los componentes más importantes son los numeros:" , auto 

#print (porcentajemayores(val)) 
     
#las componentes más importantes son las dos primeras-

#print(np.transpose(data1))
Pr= (np.array([vec[0],vec[1]]))
T = np.dot(Pr,np.transpose(data1)) #Proyeccion 

#guardar las imagenes de las proyecciones 

plt.scatter(T[0,:],T[1,:])
plt.savefig("MelendezLaura_PCA.pdf")
plt.show()

print( "Se muestra como PCA puede ayudar a la identificación de patrones que facilitarán posibles agrupamientos posteriores de tumores que podrían ser tediosos de otras formas" )































