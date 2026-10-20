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
  if A.shape != B.shape:
    return False
  else:
    for i in range(A.shape[0]):
      for j in range(A.shape[1]):
        if abs(A[i][j] - B[i][j]) > np.finfo('float32').eps:
          return False
  return True
