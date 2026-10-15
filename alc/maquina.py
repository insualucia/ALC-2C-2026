"""L01 - Numeros de maquina."""
import numpy as np


def error(x, y):
    """
    Recibe dos numeros x e y, y calcula el error de aproximar x usando y en float64
    """
    y = np.float64(y)
    return abs(x - y)


def error_relativo(x, y):
    """
    Recibe dos numeros x e y, y calcula el error relativo de aproximar x usando y en float64
    """
    y = np.float64(y)
    return abs((x - y) / x)


def matricesIguales(A, B):
    """
    Devuelve True si ambas matrices son iguales y False en otro caso.
    Considerar que las matrices pueden tener distintas dimensiones, ademas de distintos valores.
    """
    raise NotImplementedError("Implementar matricesIguales")
