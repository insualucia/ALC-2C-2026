"""L06 - Metodo de la potencia y diagonalizacion."""
import numpy as np

from .basicos import (
    calcularAx, esSimetrica, multiplicacionDeMatrices, productoExterno,
    productoInterno, traspuesta,
)
from .normas import norma


def transformacion_de_f(A, v):

    w = calcularAx(A, v)
    norm = np.linalg.norm(w, 2)
    if norm == 0:
        return 0
    return w / np.linalg.norm(w, 2)


def metpot2k(A, tol=1e-15, K=1000):
    """
    A una matriz de n x n
    tol la tolerancia en la diferencia entre un paso y el siguiente de la estimacion del autovector.
    K el numero maximo de iteraciones a realizarse.
    Retorna vector v, autovalor lambda y numero de iteracion realizadas k
    """
    v = np.zeros(A.shape[1])
    while np.all(v == 0):
        v = np.random.rand(A.shape[1])

    v_monio = transformacion_de_f(A, transformacion_de_f(A, v))
    e = productoInterno(v_monio, v)
    k = 0

    while np.abs(e - 1) > tol and k < K:
        v = v_monio
        v_monio = transformacion_de_f(A, transformacion_de_f(A, v))
        e = productoInterno(v_monio, v)
        k += 1

    l = productoInterno(v_monio, calcularAx(A, v_monio))

    return v_monio, l, k


def diagRH(A, tol=1e-15, K=1000):
    """
    A una matriz simetrica de n x n
    tol la tolerancia en la diferencia entre un paso y el siguiente de la estimacion del autovector.
    K el numero maximo de iteraciones a realizarse
    retorna matriz de autovectores S y matriz de autovalores D, tal que A = S D S.T
    Si la matriz A no es simetrica, debe retornar None
    """
    if not esSimetrica(A):
        return None

    n = A.shape[0]
    if n == 1:
        return np.eye(1), np.array(A, dtype=float)

    v1, l1, _ = metpot2k(A, tol, K)

    # reflector de Householder que lleva v1 a e1
    w = -v1
    w[0] += 1
    norma_w = norma(w, 2)
    if norma_w < tol:
        # v1 ya es e1, no hace falta reflejar
        H = np.eye(n)
    else:
        H = np.eye(n) - 2 * productoExterno(w, w) / norma_w ** 2

    B = multiplicacionDeMatrices(multiplicacionDeMatrices(H, A), traspuesta(H))

    if n == 2:
        return H, B

    # B es simetrica en teoria, pero no exactamente por el redondeo: sin esto
    # la llamada recursiva devolveria None
    A_monio = B[1:, 1:]
    A_monio = (A_monio + traspuesta(A_monio)) / 2
    S_monio, D_monio = diagRH(A_monio, tol, K)

    D = np.zeros((n, n))
    D[0, 0] = l1
    D[1:, 1:] = D_monio

    S_bloque = np.zeros((n, n))
    S_bloque[0, 0] = 1
    S_bloque[1:, 1:] = S_monio
    S = multiplicacionDeMatrices(H, S_bloque)

    return S, D
