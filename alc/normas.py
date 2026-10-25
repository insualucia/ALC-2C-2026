"""L03 - Normas y numero de condicion."""
import numpy as np

from .basicos import calcularAx
from .lu import inversa


def norma(x, p):
    """
    Devuelve la norma p del vector x.
    """
    if p == 'inf':
        return float(np.max(np.abs(x)))
    return float(np.pow(np.sum(np.pow(np.abs(x), p)), (1 / p)))


def normaliza(X, p):
    """
    Recibe X, una lista de vectores no vacios, y un escalar p. Devuelve
    una lista donde cada elemento corresponde a normalizar los
    elementos de X con la norma p.
    """
    vectores_normalizados = []
    for x in X:
        n = norma(x, p)
        if n == 0:
            vectores_normalizados.append(x)
        else:
            vectores_normalizados.append(x / n)
    return vectores_normalizados


def normaMatMC(A, q, p, Np):
    """
    Devuelve la norma ||A||_{q,p} y el vector x en el cual se alcanza
    el maximo.
    """
    n = A.shape[1]
    vectores_aleatorios = [np.random.randn(n) for i in range(Np)]
    vectores_normalizados = normaliza(vectores_aleatorios, p)
    normas_transformadas = [norma(calcularAx(A, x), q) for x in vectores_normalizados]

    indice_max = np.argmax(normas_transformadas)
    norma_max = normas_transformadas[indice_max]
    x_max = vectores_normalizados[indice_max]
    return norma_max, x_max


def normaExacta(A, p=[1, 'inf']):
    """
    Devuelve una lista con las normas 1 e infinito de una matriz A
    usando las expresiones del enunciado 2.(c).
    """
    if p == 1:
        return np.max(np.sum(np.abs(A), 0))
    elif p == 'inf':
        return np.max(np.sum(np.abs(A), 1))
    elif p == [1, 'inf']:
        return [normaExacta(A, 1), normaExacta(A, 'inf')]
    else:
        return None


def condMC(A, p, Np=10000):
    """
    Devuelve el numero de condicion de A usando la norma inducida p.
    """
    A_inv = inversa(A)
    return normaMatMC(A, p, p, Np)[0] * normaMatMC(A_inv, p, p, Np)[0]


def condExacto(A, p):
    """
    Que devuelve el numero de condicion de A a partir de la formula de
    la ecuacion (1) usando la norma p.
    """
    return normaExacta(A, p) * normaExacta(inversa(A), p)
