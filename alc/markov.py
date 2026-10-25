"""L07 - Cadenas de Markov y matrices ralas."""
import numpy as np
from numpy.ma.extras import column_stack

from alc import multiplicacionDeMatrices, diagRH, traspuesta


def transiciones_al_azar_continuas(n):
    """
    n la cantidad de filas (columnas) de la matriz de transicion.
    Retorna matriz T de n x n normalizada por columnas, y con entradas al azar en el intervalo [0,1]
    """

    m = np.random.rand(n,n)
    for j in range(n):
        col_j = m[:,j]
        sum_j = sum(col_j)
        for i in range(n):
            col_j[i] = col_j[i]/sum_j

    return m


def transiciones_al_azar_uniformes(n, thres):
    """
    n la cantidad de filas (columnas) de la matriz de transicion.
    thres probabilidad de que una entrada sea distinta de cero.
    Retorna matriz T de n x n normalizada por columnas.
    El elemento i,j es distinto de cero si el numero generado al azar para i,j es menor o igual a thres.
    Todos los elementos de la columna j son iguales
    (a 1 sobre el numero de elementos distintos de cero en la columna).
    """
    m = (np.random.rand(n, n) <= thres).astype(float)
    for j in range(n):
        sum_j = sum(m[:, j])
        if sum_j == 0:
            m[j, j] = 1
        else:
            m[:, j] = m[:, j] / sum_j
    return m


def nucleo(A, tol=1e-15):
    """
    A una matriz de m x n
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Calcula el nucleo de la matriz A diagonalizando la matriz traspuesta(A) * A (* la multiplicacion matricial),
    usando el medodo diagRH. El nucleo corresponde a los autovectores de autovalor con modulo <= tol.
    Retorna los autovectores en cuestion, como una matriz de n x k, con k el numero de autovectores en el nucleo.
    """

    AtA = multiplicacionDeMatrices(traspuesta(A),A)
    S, D = diagRH(AtA)
    autovectores = []
    for j in range(len(AtA)):
        if D[j,j] <= tol:
            autovectores.append(S[:,j])
    if len(autovectores) == 0:
        return np.array([])
    else:
        return np.column_stack(autovectores)


def crea_rala(listado, m_filas, n_columnas, tol=1e-15):
    """
    Recibe una lista listado, con tres elementos: lista con indices i, lista con indices j, y lista con valores
    A_ij de la matriz A. Tambien las dimensiones de la matriz a traves de m_filas y n_columnas.
    Los elementos menores a tol se descartan.
    Idealmente, el listado debe incluir unicamente posiciones correspondientes a valores distintos de cero.
    Retorna una lista con:
    - Diccionario {(i,j):A_ij} que representa los elementos no nulos de la matriz A. Los elementos con modulo
      menor a tol deben descartarse por default.
    - Tupla (m_filas,n_columnas) que permita conocer las dimensiones de la matriz.
    """
    dicc = {}
    if len(listado) == 0:
        return [dicc, (m_filas, n_columnas)]

    filas, columnas, valores = listado
    for k in range(len(valores)):
        if abs(valores[k]) >= tol:
            dicc[(filas[k], columnas[k])] = valores[k]

    return [dicc, (m_filas, n_columnas)]


def multiplica_rala_vector(A, v):
    """
    Recibe una matriz rala creada con crea_rala y un vector v.
    Retorna un vector w resultado de multiplicar A con v
    """
    dicc, (m_filas, n_columnas) = A
    w = np.zeros(m_filas)
    for (i, j), a_ij in dicc.items():
        w[i] += a_ij * v[j]

    return w