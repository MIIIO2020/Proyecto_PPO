# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 17:47:51 2020

@author: rolft
"""

import numpy as np
from numpy import random
import collections
import time

# Información problema
S=50       #capacidad max y tamaño de lote
a=0       #Demanda minima
b=30      #Demanda maxima
cost=300  #precio cerveza y costo perdida clientes
k=100     #costo de realizar el pedido  
price=1500
tasa=1#0.9999

# =============================================================================
# Generó las demandas
# =============================================================================
random.seed(10)
total=50000
x=random.randint(a,b+1, size=(total))
Dem_frec = collections.Counter(x)

# =============================================================================
# Comiezo a generar mi matriz de probaibilidades
# =============================================================================
prob =np.zeros([S+1,S+1])
for i in range(S+1):
    for j in range(S+1):
        dem=i-j
        if(S>=dem>=0):
            if(dem<i):
               prob[i][j] =Dem_frec[dem]/total
    prob[i][0] = 1 - sum(prob[i][0:])


# =============================================================================
# Expected Regard Vector
# =============================================================================
# aquí se ajustan los costos de
# D>i o D<i

reward=np.zeros(S+1)
for i in range(S+1):
    r=0
    for dem in range(a,b+1):
        p=Dem_frec[dem]/total
        r_dem=price*min(dem,i)
        if(dem>=i):
            r_dem+=-(p-cost)*(dem-i)
        else:
            #if(dem<i):#Agrego costo de inventario sobrante
            r_dem+= -(cost)*(i-dem)
        r+= p*(r_dem)
    #print(r)
    reward[i]=r


# =============================================================================
# Value Determination Equations
# =============================================================================
#funcion para obtener el valor de la ecuaciones
def Value_Determination_Equations(decisions,prob,descuento,costo,revenues,S,k):
    L=len(decisions)
    aux=np.zeros([L,L])
    B=np.zeros(L)
    for i in range(L):
        cant_comprada_i=decisions[i]
        if cant_comprada_i>0:
        #Juan decide comprar mas cerveza.
            if(cant_comprada_i+i<=S):
            #La compra más el inventario debe ser menor a Cap max
                B[i]=revenues[cant_comprada_i+i]
                B[i]+=-costo*(cant_comprada_i)-k
                aux[i]=prob[cant_comprada_i+i]
            else:
                B[i]+=-costo*(cant_comprada_i)*10000 -k#Se utliza una big M 
                aux[i]=prob[i]
        else:
            aux[i]=prob[i]
            B[i]=revenues[i]
    aux = np.insert(aux, aux.shape[1], np.zeros(L), 1)
    aux = np.insert(aux, aux.shape[0], np.zeros(L+1), 0)
    A = np.identity(L+1)-(descuento*aux)
    A[:,L] = 1
    B = np.append(B, 0)
    X = np.linalg.solve(A,B)
    X = X[:-1]
    return X

# =============================================================================
# Howard’s Policy Iteration Method
# =============================================================================
# =============================================================================
# 
# =============================================================================
def T_delta(prob,values,descuento,costo,revenues,k):
    L=len(values)
    T=np.zeros(L)
    Cant=np.zeros(L)
    for i in range(L):
        lista=[]
        lista_2=[]
        for j in range(i,L):
            if(j>i):#si se realiza el pedido hay un costo
                lista.append(revenues[j]-costo*(j-i)-k +(descuento*sum(prob[j]*values)))
                lista_2.append(j-i)
            else:# No realizo un pedido
                lista.append(revenues[j] +(descuento*sum(prob[j]*values)))
                lista_2.append(j-i)
            
        T[i]=max(lista)#Con i=S soy indiferente a comprar o no
        indice=lista.index(T[i])
        Cant[i]=lista_2[indice]
    return [T,Cant]

# =============================================================================
# Busqueda del Optimo
# =============================================================================
def Search_OP(decisions,prob,descuento,costo,revenues,S,k):
   
    #(D,m_prob,descuento,c,states_reveneus)
    V_D_E = Value_Determination_Equations(decisions,prob,
                                          descuento,costo,revenues,S,k)
    Los_T, D_T = T_delta(prob,V_D_E,descuento,costo,revenues,k)
    val_Op=np.around(Los_T,4)==np.around(V_D_E,4)
    if(sum(val_Op)==len(val_Op)):
        return(decisions)
    else:
        for i in range(len(val_Op)):
            if (val_Op[i]!=True):
                decisions[i]=D_T[i].astype('int')
        return(Search_OP(decisions,prob,descuento,costo,revenues,S,k))


# =============================================================================
# Busqueda del Optimo
# =============================================================================
def while_Search_OP(decisions,prob,descuento,costo,revenues,S,k):
    while(True):
        
        #(D,m_prob,descuento,c,states_reveneus)
        V_D_E = Value_Determination_Equations(decisions,prob,
                                              descuento,costo,revenues,S,k)
        Los_T, D_T = T_delta(prob,V_D_E,descuento,costo,revenues,k)
        val_Op=np.around(Los_T,4)==np.around(V_D_E,4)
        if(sum(val_Op)==len(val_Op)):
            return(decisions)
        else:
            for i in range(len(val_Op)):
                if (val_Op[i]!=True):
                    decisions[i]=D_T[i].astype('int')

# =============================================================================
# Fin de Funciones
# =============================================================================
D=np.ones(S+1).astype('int')
start_time = time.time()
Optimo=Search_OP(D,prob,tasa,cost,reward,S,k)
print("--- %s Segundos ---" % (time.time() - start_time))
print(Optimo)