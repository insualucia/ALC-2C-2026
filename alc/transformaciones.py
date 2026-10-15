"""L02 - Transformaciones lineales."""
import numpy as np

from .basicos import calcularAx, multiplicacionDeMatrices


def rota(theta):
    """
    Recibe un angulo theta y retorna una matriz de 2x2
    que rota un vector dado en un angulo theta
    """
    T = np.array([
        [np.cos(theta), -1 * np.sin(theta)],
        [np.sin(theta), np.cos(theta)]])
    return T


def escala(s):
    """
    Recibe una tira de numeros s y retorna una matriz cuadrada de
    n x n, donde n es el tamano de s.
    La matriz escala la componente i de un vector de Rn
    en un factor s[i]
    """
    A = np.zeros([len(s), len(s)])
    for i in range(len(s)):
        A[i][i] = s[i]
    return A


def rota_y_escala(theta, s):
    """
    Recibe un angulo theta y una tira de numeros s,
    y retorna una matriz de 2x2 que rota el vector en un angulo theta
    y luego lo escala en un factor s
    """
    E = escala(s)
    R = rota(theta)
    assert R.shape[1] == E.shape[0]
    return multiplicacionDeMatrices(E, R)


def afin(theta, s, b):
    """
    Recibe un angulo theta, una tira de numeros s (en R2), y un vector b en R2.
    Retorna una matriz de 3x3 que rota el vector en un angulo theta,
    luego lo escala en un factor s y por ultimo lo mueve en un valor fijo b
    """
    RyE = rota_y_escala(theta, s)
    A = np.zeros([3, 3])
    for i in range(len(RyE)):
        for j in range(len(RyE[i])):
            A[i][j] = RyE[i][j]
    A[0][2] = b[0]
    A[1][2] = b[1]
    A[2][2] = 1
    return A


def trans_afin(v, theta, s, b):
    """
    Recibe un vector v (en R2), un angulo theta,
    una tira de numeros s (en R2), y un vector b en R2.
    Retorna el vector w resultante de aplicar la transformacion afin a v
    """
    F = afin(theta, s, b)
    # v en coordenadas homogeneas: (v0, v1, 1)
    w = calcularAx(F, np.array([v[0], v[1], 1]))
    return w[:2]
