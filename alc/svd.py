"""L08 - Descomposición en Valores Singulares."""
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


