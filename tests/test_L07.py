"""Tests de L07 (Markov y matrices ralas). Original: material/L07/L07-Tests.py"""
import numpy as np

from alc import (
    transiciones_al_azar_continuas, transiciones_al_azar_uniformes,
    nucleo, crea_rala, multiplica_rala_vector,
)


# ---------------------------------------------------------------------
# Auxiliares provistas por la catedra
# ---------------------------------------------------------------------

def es_markov(T,tol=1e-6):
    """
    T una matriz cuadrada.
    tol la tolerancia para asumir que una suma es igual a 1.
    Retorna True si T es una matriz de transición de Markov (entradas no negativas y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
    """
    n = T.shape[0]
    for i in range(n):
        for j in range(n):
            if T[i,j]<0:
                return False
    for j in range(n):
        suma_columna = sum(T[:,j])
        if np.abs(suma_columna - 1) > tol:
            return False
    return True

def es_markov_uniforme(T,thres=1e-6):
    """
    T una matriz cuadrada.
    thres la tolerancia para asumir que una entrada es igual a cero.
    Retorna True si T es una matriz de transición de Markov uniforme (entradas iguales a cero o iguales entre si en cada columna, y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
    """
    if not es_markov(T,thres):
        return False
    # cada columna debe tener entradas iguales entre si o iguales a cero
    m = T.shape[1]
    for j in range(m):
        non_zero = T[:,j][T[:,j] > thres]
        # all close
        close = all(np.abs(non_zero - non_zero[0]) < thres)
        if not close:
            return False
    return True


def esNucleo(A,S,tol=1e-5):
    """
    A una matriz m x n
    S una matriz n x k
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Retorna True si las columnas de S estan en el nucleo de A (es decir, A*S = 0. Esto no chequea si es todo el nucleo
    """
    for col in S.T:
        res = A @ col
        if not np.allclose(res,np.zeros(A.shape[0]), atol=tol):
            return False
    return True


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_transiciones_al_azar():
    for i in range(1,100):
        T = transiciones_al_azar_continuas(i)
        assert es_markov(T), f"transiciones_al_azar_continuas fallo para n={i}"

        T = transiciones_al_azar_uniformes(i,0.3)
        assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
        # Si no atajan casos borde, pueden fallar estos tests. Recuerden que suma de columnas DEBE ser 1, no valen columnas nulas.
        T = transiciones_al_azar_uniformes(i,0.01)
        assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
        T = transiciones_al_azar_uniformes(i,0.01)
        assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"


def test_nucleo():
    A = np.eye(3)
    S = nucleo(A)
    assert S.shape[0]==0, "nucleo fallo para matriz identidad"
    A[1,1] = 0
    S = nucleo(A)
    msg = "nucleo fallo para matriz con un cero en diagonal"
    assert esNucleo(A,S), msg
    assert S.shape==(3,1), msg
    assert abs(S[2,0])<1e-2, msg
    assert abs(S[0,0])<1e-2, msg

    v = np.random.random(5)
    v = v / np.linalg.norm(v)
    H = np.eye(5) - np.outer(v, v)  # proyección ortogonal
    S = nucleo(H)
    msg = "nucleo fallo para matriz de proyeccion ortogonal"
    assert S.shape==(5,1), msg
    v_gen = S[:,0]
    v_gen = v_gen / np.linalg.norm(v_gen)
    assert np.allclose(v, v_gen) or np.allclose(v, -v_gen), msg


def test_crea_rala():
    listado = [[0,17],[3,4],[0.5,0.25]]
    A_rala_dict, dims = crea_rala(listado,32,89)
    assert dims == (32,89), "crea_rala fallo en dimensiones"
    assert A_rala_dict[(0,3)] == 0.5, "crea_rala fallo"
    assert A_rala_dict[(17,4)] == 0.25, "crea_rala fallo"
    assert len(A_rala_dict) == 2, "crea_rala fallo en cantidad de elementos"

    listado = [[32,16,5],[3,4,7],[7,0.5,0.25]]
    A_rala_dict, dims = crea_rala(listado,50,50)
    assert dims == (50,50), "crea_rala fallo en dimensiones con tol"
    assert A_rala_dict.get((32,3)) == 7
    assert A_rala_dict[(16,4)] == 0.5
    assert A_rala_dict[(5,7)] == 0.25

    listado = [[1,2,3],[4,5,6],[1e-20,0.5,0.25]]
    A_rala_dict, dims = crea_rala(listado,10,10)
    assert dims == (10,10), "crea_rala fallo en dimensiones con tol"
    assert (1,4) not in A_rala_dict
    assert A_rala_dict[(2,5)] == 0.5
    assert A_rala_dict[(3,6)] == 0.25
    assert len(A_rala_dict) == 2

    # caso borde: lista vacia. Esto es una matriz de 0s
    listado = []
    A_rala_dict, dims = crea_rala(listado,10,10)
    assert dims == (10,10), "crea_rala fallo en dimensiones con lista vacia"
    assert len(A_rala_dict) == 0, "crea_rala fallo en cantidad de elementos con lista vacia"


def test_multiplica_rala_vector():
    listado = [[0,1,2],[0,1,2],[1,2,3]]
    A_rala = crea_rala(listado,3,3)
    v = np.random.random(3)
    v = v / np.linalg.norm(v)
    res = multiplica_rala_vector(A_rala,v)
    A = np.array([[1,0,0],[0,2,0],[0,0,3]])
    res_esperado = A @ v
    assert np.allclose(res,res_esperado), "multiplica_rala_vector fallo"

    A = np.random.random((5,5))
    A = A * (A > 0.5) 
    listado = [[],[],[]]
    for i in range(5):
        for j in range(5):
            listado[0].append(i)
            listado[1].append(j)
            listado[2].append(A[i,j])

    A_rala = crea_rala(listado,5,5)
    v = np.random.random(5)
    assert np.allclose(multiplica_rala_vector(A_rala,v), A @ v)
