"""L04 - Descomposicion LU."""
import numpy as np


def calculaLU(A):
    """
    Descomposicion LU sin pivoteo (llama a elim_gaussiana).
    Parametros: A: matriz cuadrada (array 2D).
    Retorna: L, U (triangular inferior con unos en la diagonal y triangular superior) y la cantidad de operaciones.
    """
    return elim_gaussiana(A)


def res_tri(L, b, inferior):
    """
    Resuelve un sistema triangular L x = b por sustitucion hacia adelante o hacia atras.
    Parametros: L: matriz triangular cuadrada. b: vector del lado derecho. inferior: True si L es triangular inferior, False si es superior.
    Retorna: El vector solucion x.
    """
    n = L.shape[1]
    x = np.zeros(n)

    if inferior:
        for i in range(n):
            suma = 0
            for j in range(i):
                suma += L[i, j] * x[j]
            x[i] = (b[i] - suma) / L[i, i]

    else:
        for i in range(n-1, -1, -1):
            suma = 0
            for j in range(i+1, n):
                suma += L[i, j] * x[j]
            x[i] = (b[i] - suma) / L[i, i]
    return x

def inversa(A):
    """
    Inversa de una matriz cuadrada, resolviendo A x = e_i para cada columna con la descomposicion LU.
    Parametros: A: matriz cuadrada inversible (array 2D).
    Retorna: La matriz inversa de A.
    """
    n = A.shape[0]

    L, U, cant_op = elim_gaussiana(A)

    invA = np.zeros((n, n))

    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        y = res_tri(L, e, True)
        x = res_tri(U, y, False)
        invA[:, i] = x
    return invA

def traspuesta(A):
    """
    Traspuesta de una matriz cuadrada (la version general esta en basicos.py; en alc.py se usa esa).
    Parametros: A: matriz cuadrada (array 2D).
    Retorna: Una matriz nueva con B[j, i] = A[i, j].
    """
    n = A.shape[1]
    B = A.copy()
    for i in range(n):
        for j in range(n):
            B[j, i] = A[i, j]
    return B 


def calculaLDV (A):
    """
    Factorizacion A = L D V, con L y V triangulares con unos en la diagonal y D diagonal, a partir de dos eliminaciones gaussianas (la segunda sobre U^T).
    Parametros: A: matriz cuadrada con LU sin pivoteo (array 2D).
    Retorna: L, D, V.
    """
    L, U, cant_op = elim_gaussiana(A)
    U_traspuesta = traspuesta(U)
    V_traspuesta, D, cant_op1 = elim_gaussiana(U_traspuesta)
    V = traspuesta(V_traspuesta)
    return L, D, V

def elim_gaussiana(A):
    """
    Eliminacion gaussiana sin pivoteo: A = L U.
    Parametros: A: matriz cuadrada (array 2D); no se modifica.
    Retorna: L (triangular inferior con unos en la diagonal), U (triangular superior) y la cantidad de operaciones realizadas. Si A no es cuadrada imprime un aviso y devuelve None.
    """
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.astype(float)
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    ## desde aqui -- CODIGO A COMPLETAR
    for k in range(n-1):
        for i in range(k+1, n):
            multiplicador = Ac[i, k] / Ac[k, k]
            cant_op += 1
            Ac[i, k] = multiplicador
            for j in range(k + 1, n):
                Ac[i, j] = Ac[i,j] - multiplicador * Ac[k, j]
                cant_op += 2

    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i > j:
                L[i, j] = Ac[i, j]
            else:
                U[i, j] = Ac[i, j]

                
    ## hasta aqui, calculando L, U y la cantidad de operaciones sobre 
    ## la matriz Ac
            
    
    return L, U, cant_op

def esSDP(A, atol=1e-10):
    """
    Indica si una matriz es simetrica definida positiva: es simetrica (con tolerancia) y todos los pivotes D[i, i] de su LDV son > atol.
    Parametros: A: matriz cuadrada (array 2D). atol: tolerancia.
    Retorna: True si es simetrica definida positiva, False si no.
    """
    n = A.shape[1]
    for i in range(n):
        for j in range(n):
            if abs(A[i, j] - A[j, i]) > atol:
                return False
    L, D, V = calculaLDV(A)
    for i in range(n):
        if D[i, i] <= atol:
            return False
    return True
            
