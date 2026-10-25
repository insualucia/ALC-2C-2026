# Tests L08
import numpy as np
from .basicos import calcularAx, productoExterno, productoInterno, traspuesta, multiplicacionDeMatrices
from .autovalores import diagRH
from .qr import QR_con_HH

def svd_reducida(A, k="max", tol=1e-15):
    """
    A la matriz de interes (de m x n)
    k es el numero de valores singulares (y vectores) a retener.
    tol es la tolerancia para considerar un valor singular igual a cero.
    Retorna hatU (m x k), hatSig (vector de k valores singulares) y hatV (n x k).
    """
    A = np.asarray(A, dtype=float)
    m = A.shape[0]
    n = A.shape[1]

    if m < n:                       
        U, S, V = svd_reducida(A.T, k, tol)
        return V, S, U

    S_aut, D = diagRH(multiplicacionDeMatrices(traspuesta(A), A))

    lam = np.zeros(n)
    for i in range(n):
        lam[i] = D[i, i]
    V = np.array(S_aut)

    # ordeno de mayor a menor, permutando columnas de V
    for i in range(n):
        imax = i
        for j in range(i + 1, n):
            if lam[j] > lam[imax]:
                imax = j
        if imax != i:
            aux = lam[i]; lam[i] = lam[imax]; lam[imax] = aux
            col = np.array(V[:, i]); V[:, i] = V[:, imax]; V[:, imax] = col
    S = np.zeros(n)
    for i in range(n):
        if lam[i] > 0:
            S[i] = np.sqrt(lam[i])

    # cuantos retener: sigma > tol y lambda no es solo ruido de redondeo
    r = 0
    for i in range(n):
        if S[i] > tol and lam[i] > 1e-12 * lam[0]:
            r += 1
    if k != "max":
        r = min(r, int(k))

    # u_i = A v_i / sigma_i
    AV = multiplicacionDeMatrices(A, V[:, :r])
    hatU = np.zeros((m, r))
    for i in range(r):
        hatU[:, i] = AV[:, i] / S[i]

    return hatU, S[:r], V[:, :r]

# Matrices al azar
def genera_matriz_para_test(m,n=2,tam_nucleo=0):
    if tam_nucleo == 0:
        A = np.random.random((m,n))
    else:
        A = np.random.random((m,tam_nucleo))
        A = np.hstack([A,A])
    return(A)

def test_svd_reducida_mn(A,tol=1e-15):
    m,n = A.shape
    hU,hS,hV = svd_reducida(A,tol=tol)
    nU,nS,nVT = np.linalg.svd(A)
    r = len(hS)+1
    assert np.all(np.abs(np.abs(np.diag(hU.T @ nU))-1)<10**r*tol), 'Revisar calculo de hat U en ' + str((m,n))
    assert np.all(np.abs(np.abs(np.diag(nVT @ hV))-1)<10**r*tol), 'Revisar calculo de hat V en ' + str((m,n))
    assert len(hS) == len(nS[np.abs(nS)>tol]), 'Hay cantidades distintas de valores singulares en ' + str((m,n))
    assert np.all(np.abs(hS-nS[np.abs(nS)>tol])<10**r*tol), 'Hay diferencias en los valores singulares en ' + str((m,n))

for m in [2,5,10,20]:
    for n in [2,5,10,20]:
        for _ in range(10):
            A = genera_matriz_para_test(m,n)
            test_svd_reducida_mn(A)


# Matrices con nucleo

m = 12
for tam_nucleo in [2,4,6]:
    for _ in range(10):
        A = genera_matriz_para_test(m,tam_nucleo=tam_nucleo)
        test_svd_reducida_mn(A)

# Tamaños de las reducidas
A = np.random.random((8,6))
for k in [1,3,5]:
    hU,hS,hV = svd_reducida(A,k=k)
    assert hU.shape[0] == A.shape[0], 'Dimensiones de hU incorrectas (caso a)'
    assert hV.shape[0] == A.shape[1], 'Dimensiones de hV incorrectas(caso a)'
    assert hU.shape[1] <= k, 'Dimensiones de hU incorrectas (caso a)'
    assert hV.shape[1] <= k, 'Dimensiones de hV incorrectas(caso a)'
    assert len(hS) <= k, 'Tamaño de hS incorrecto'
