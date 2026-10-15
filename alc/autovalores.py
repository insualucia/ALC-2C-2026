"""L06 - Metodo de la potencia y diagonalizacion (EN PROGRESO)."""
import numpy as np

from .basicos import calcularAx, esSimetrica, productoInterno


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
    while v == 0:
        v = np.random.rand(A.shape[1])

    v_monio = transformacion_de_f(A, v)
    e = productoInterno(v_monio, v)
    k = 0

    while np.abs(e - 1) > tol and k < K:
        v = v_monio
        v_monio = transformacion_de_f(A, transformacion_de_f(A, v))
        e = productoInterno(v_monio, v)
        k += 1

    l = productoInterno(v_monio, calcularAx(A, v_monio))

    return l, v_monio, k


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

    # TODO: resolver
