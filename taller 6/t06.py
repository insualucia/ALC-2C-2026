import numpy as np
from modulosALC import multiplicacionDeMatrices, multiplicacionDeVectores
from librerias import simetrica

def transformacion_de_f(A, v):

    w = multiplicacionDeMatrices(A, v)
    norm = np.linalg.norm(w, 2)
    if norm == 0:
        return 0
    return w / np.linalg.norm(w, 2)

def metpot2k(A, tol=1e-15, K=1000):

    v = np.zeros(A.shape[1])
    while v == 0:
        v = np.random.rand(A.shape[1])

    v_monio = transformacion_de_f(A, v)
    e = multiplicacionDeVectores(v_monio, v)
    k = 0

    while np.abs(e - 1) > tol and k < K:
        v = v_monio
        v_monio = transformacion_de_f(A, transformacion_de_f(A, v))
        e = multiplicacionDeVectores(v_monio, v)
        k += 1

    l = multiplicacionDeVectores(v_monio, multiplicacionDeMatrices(A, v_monio))

    return l, v_monio, k

def diagRH(A, tol=1e-15, K=1000):

    if not simetrica(A):
        return None

    
    
        