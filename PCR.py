import numpy as np

#extraer los datos

data = (np.genfromtxt('WDBC.dat', delimiter = ','))[:,1:]
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
          
#print(matrizcov(data))

A = matrizcov(data)

val,vec = np.linalg.eig(A)

print(val)

#calcular los autovectores de sus autovalores

#for i in range(np.shape(val)[0]):
    #print("el vector", vec[:,i], "corresponde al valor" , val[i])
    

#hallaremos los parametros más importantes en medida de sus porcentajes

def porcentajemayores(array):  
    total = np.sum(array)
    narray = np.array([])
    nvalores = np.array([])
    for i in range(np.shape(array)[0]):
        totali = (array[i] * 100)/total
        if(totali>=15): #tocó sobre el 15% para escoger dos 
             narray = np.append(narray,totali)
             nvalores = np.append(nvalores,array[i]) 


    return narray , nvalores 
suma = np.sum(val)
print (porcentajemayores(val))      








