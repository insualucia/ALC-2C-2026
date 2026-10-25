"""L08 - Descomposición en Valores Singulares."""
import numpy as np
from .basicos import calcularAx, productoExterno, productoInterno, traspuesta, multiplicacionDeMatrices
from .autovalores import diagRH
def svd_reducida(A, k="max", tol=1e-15):
    """
    A la matriz de interes (de m x n)
    k es el numero de valores singulares (y vectores) a retener.
    tol es la tolerancia para considerar un valor singular igual a cero.
    Retorna hatU (matriz de m x k), hatSig (vector de k valores singulares) y hatV (matriz de n x k).
    """
    A = np.array(A, dtype=float)
    m, n = A.shape

    if m < n:
        U, S, V = svd_reducida(traspuesta(A), k, tol)
        return V, S, U
    autovalores, autovectores = diagRH(multiplicacionDeMatrices(traspuesta(A), A))
    return None
