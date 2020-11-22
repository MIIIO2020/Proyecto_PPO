import numpy as np

# =============================================================================
# Datos
# =============================================================================
#next period stats Probabilities
m_prob=np.array([[0.7,0.3,0,0],
                [0,0.7,0.3,0],
                [0,0,0.6,0.4],
                [0,0,0,1]]
                )
#Reveneus por estados
states_reveneus=np.array([100,80,50,10])
c=-100

# =============================================================================
#  Value Determination Equations
# =============================================================================
def Value_Determination_Equations(decisions,prob,descuento,costo,revenues):
    L=len(decisions)
    aux=np.zeros([L,L])
    B=np.zeros(L)
    for i in range(L):
        if decisions[i]:
            aux[i]=prob[0]
            B[i]=costo
        else:
            aux[i]=prob[i]
            B[i]=revenues[i]
    A = np.identity(L)-(descuento*aux)
    X = np.linalg.solve(A,B)
    return X

# =============================================================================
# Howard’s Policy Iteration Method
# =============================================================================
D=np.array([0,0,1,1])
V_D_E=Value_Determination_Equations(D,m_prob,0.9,c,states_reveneus)


def T_delta(prob,values,descuento,costo,revenues):
    L=len(values)
    T=np.zeros(L)
    #aux_1 siempre igual
    aux_1=costo+(descuento*sum(prob[0]*values))
    for i in range(L):
        aux_2=revenues[i]+(descuento*sum(prob[i]*values))
        T[i]=max(aux_1,aux_2)
    return T
# =============================================================================
# Función Recursiva que encuentra la politica optima
# =============================================================================
def Search_OP(decisions,prob,descuento,costo,revenues):
    print('recurción')
    print(decisions)
    #(D,m_prob,descuento,c,states_reveneus)
    V_D_E=Value_Determination_Equations(decisions,prob,descuento,costo,
                                        revenues)
    
    Los_T=T_delta(prob,V_D_E,descuento,costo,revenues)
    val_Op=np.around(Los_T,4)==np.around(V_D_E,4)
    print(val_Op)
    if(sum(val_Op)==len(val_Op)):
        return(decisions)
    else:
        for i in range(len(val_Op)):
            if (val_Op[i]!=True):
                if(decisions[i]==1):
                    decisions[i]=0
                else:
                    decisions[i]=1
        #For
        return(Search_OP(decisions,prob,descuento,costo,revenues))

D=np.array([1,1,1,1])
D_optimo=Search_OP(D,m_prob,0.9,c,states_reveneus)
print('Las politicas optimas son',D_optimo)
