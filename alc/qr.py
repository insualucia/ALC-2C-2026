"""L05 - Factorizacion QR (Gram-Schmidt y Householder)."""
import numpy as np

from .basicos import calcularAx, productoExterno, productoInterno, traspuesta
from .normas import norma


def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    """
    A una matriz de n x n
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A no es de n x n, debe retornar None
    """
    # TODO: retorna_nops todavia no esta implementado
    n = A.shape[0]
    m = A.shape[1]
    if n != m:
        return None

    Q = np.zeros((n, m))
    R = np.zeros((n, m))

    for j in range(m):
        u = A[:, j]
        for i in range(j):
            R[i, j] = productoInterno(Q[:, i], A[:, j])
            u = u - R[i, j] * Q[:, i]

        normaU = norma(u, 2)

        if normaU < tol:
            R[j, j] = 0.0
            Q[:, j] = np.zeros(n)
        else:
            R[j, j] = normaU
            Q[:, j] = u / normaU

    return Q, R


def QR_con_HH(A, tol=1e-12, extras=False):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    extras : bool, opcional
        Si es True, devuelve informacion extra sobre el proceso de factorizacion.
        Por defecto es False. Esto lo hacemos para poder graficar el proceso.
    Devuelve la factorizacion QR de A usando reflectores de Householder.
    Devuelve:
        Q, R, extra_info (si extras es True)
        Q, R (si extras es False)
    extra_info es un diccionario con la clave:
        'R_matrices': lista de las matrices R en cada paso
        'Q_matrices': lista de las matrices Q en cada paso
    """
    # TODO: extras todavia no esta implementado
    m = A.shape[0]
    n = A.shape[1]
    if n > m:
        return None

    Q = np.eye(m)
    R = np.array(A, dtype=float)

    for j in range(n):
        columnaDesdeDiag = R[j:, j]
        norma_col = norma(columnaDesdeDiag, 2)

        if norma_col >= tol:
            if columnaDesdeDiag[0] != 0:
                alfa = -np.sign(columnaDesdeDiag[0]) * norma_col
            else:
                alfa = -np.sign(columnaDesdeDiag[0]) * -norma_col

        v = columnaDesdeDiag.copy()
        v[0] -= alfa
        norma_v = norma(v, 2)

        if norma_v > tol:
            v = v / norma_v

            # v^T R  ==  R^T v
            vR = calcularAx(traspuesta(R[j:, j:]), v)
            R[j:, j:] -= 2.0 * productoExterno(v, vR)
            Qv = calcularAx(Q[:, j:], v)
            Q[:, j:] -= 2.0 * productoExterno(Qv, v)

    for i in range(m):
        for j in range(min(i, n)):
            if abs(R[i, j]) < tol:
                R[i, j] = 0.0

    return Q, R


def calculaQR(A, metodo='RH', tol=1e-12):
    """
    A una matriz de n x n
    tol la tolerancia con la que se filtran elementos nulos en R
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """
    if metodo == 'RH':
        return QR_con_HH(A, tol=tol)
    elif metodo == 'GS':
        return QR_con_GS(A, tol=tol)
    else:
        return None
