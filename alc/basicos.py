"""
Operaciones basicas sobre vectores y matrices.

- L00 (Manipulacion de matrices): funciones de librerias.py, con los nombres
  de la consigna (L00-Ejercicios.pdf).
- Productos propios (interno, externo, matriz-matriz) para no usar @,
  np.matmul ni np.dot, que la catedra no permite en el modulo ALC.

Todo el resto del modulo se apoya en este archivo, asi que no importa nada
de los demas.
"""
import numpy as np


# ---------------------------------------------------------------------
# Productos (reemplazan a @, np.matmul y np.dot)
# ---------------------------------------------------------------------

def productoInterno(u, v):
    """Devuelve <u, v> = sum_i u[i] * v[i]."""
    if len(u) != len(v):
        return None

    acc = 0
    for i in range(len(u)):
        acc = acc + u[i] * v[i]
    return acc


def productoExterno(u, v):
    """Devuelve la matriz u v^T, de len(u) x len(v)."""
    n = len(u)
    m = len(v)
    A = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            A[i, j] = u[i] * v[j]
    return A


def multiplicacionDeMatrices(A, B):
    """Devuelve A B, con A de n x k y B de k x m."""
    assert A.shape[1] == B.shape[0]
    C = np.zeros((A.shape[0], B.shape[1]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            C[i][j] = productoInterno(A[i], B[:, j])
    return C


# ---------------------------------------------------------------------
# L00 - Manipulacion de matrices
# ---------------------------------------------------------------------

# Ejercicio 1. Desarrollar una funcion esCuadrada(A) que devuelva verdadero
# si la matriz A es cuadrada y Falso en caso contrario.
def esCuadrada(A):

    for r in A:
        if r.size != len(A):
            return False

    return True

# Ejercicio 2. Desarrollar una funcion triangSup(A) que devuelva la matriz U
# correspondiente a la matriz Triangular Superior de A sin su diagonal.
def triangSup(A):

    for i in range(len(A)):
        for j in range(A[i].size):
            if i >= j:
                A[i][j] = 0

    return A

# Ejercicio 3. Desarrollar una funcion triangInf(A) que devuelva la matriz L
# correspondiente a la matriz Triangular Inferior de A sin su diagonal.
def triangInf(A):

    for i in range(len(A)):
        for j in range(A[i].size):
            if i <= j:
                A[i][j] = 0

    return A

# Ejercicio 4. Desarrollar una funcion diagonal(A) que devuelva la matriz D
# correspondiente a la matriz diagonal de A.
def diagonal(A):

    for i in range(len(A)):
        for j in range(A[i].size):
            if i != j:
                A[i][j] = 0

    return A

# Ejercicio 5. Desarrollar una funcion traza(A) que calcule la traza de una
# matriz cualquiera A.
def traza(A):

    acc = 0
    for i in range(len(A)):
        for j in range(A[i].size):
            if i == j:
                acc += A[i][j]

    return acc

# Ejercicio 6. Desarrollar una funcion traspuesta(A) que devuelva la matriz
# traspuesta de A.
def traspuesta(A):

    n = len(A)
    m = A[0].size
    B = np.zeros((m, n))
    for i in range(len(A)):
        for j in range(A[i].size):
            B[j][i] = A[i][j]

    return B

# Ejercicio 7. Desarrollar una funcion esSimetrica(A) que devuelve True si la
# matriz A es simetrica y False en caso contrario.
def esSimetrica(A):

    if not esCuadrada(A):
        return False

    for i in range(len(A)):
        for j in range(A[i].size):
            if A[i][j] != A[j][i]:
                return False
    return True

# Ejercicio 8. Desarrollar una funcion calcularAx(A,x) que recibe una matriz
# A de tamano n x m y un vector x de largo m y devuelve un vector b de largo n
# resultado de la multiplicacion vectorial de la matriz y el vector.
def calcularAx(A, x):

    if len(A[0]) != len(x):
        return None

    b = np.zeros(len(A))
    for i in range(len(A)):
        suma = 0
        for j in range(A[i].size):
            suma += A[i][j] * x[j]
        b[i] = suma

    return b

# Ejercicio 9. Desarrollar una funcion intercambiarFilas(A, i, j), que
# intercambie las filas i y la j de la matriz A. El intercambio tiene que
# ser in-place.
def intercambiarFilas(A, i, j):

    for k in range(len(A[i])):
        aux = A[i][k]
        A[i][k] = A[j][k]
        A[j][k] = aux

# Ejercicio 10. Desarrollar una funcion sumar_fila_multiplo(A, i, j, s) que
# a la fila i le sume la fila j multiplicada por un escalar s. Esta es una
# operacion elemental clave en la eliminacion gaussiana. La operacion debe
# ser in-place.
def sumar_fila_multiplo(A, i, j, s):

    fi = A[i]
    fj = A[j]
    for k in range(len(fi)):
        fi[k] += fj[k] * s

# Ejercicio 11. Desarrollar una funcion esDiagonalmenteDominante(A) que
# devuelva True si una matriz cuadrada A es estrictamente diagonalmente
# dominante. Esto ocurre si para cada fila, el valor absoluto del elemento
# en la diagonal es mayor que la suma de los valores absolutos de los demas
# elementos en esa fila.
def esDiagonalmenteDominante(A):

    for i in range(len(A)):
        d = abs(A[i][i])
        suma = 0
        for j in range(len(A[i])):
            if i != j:
                suma += abs(A[i][j])
        if d <= suma:
            return False

    return True

# Ejercicio 12. Desarrollar una funcion matrizCirculante(v) que genere una
# matriz circulante a partir de un vector. En una matriz circulante la
# primer fila es igual al vector v, y en cada fila se encuentra una
# permutacion ciclica de la fila anterior, moviendo los elementos un lugar
# hacia la derecha.
def matrizCirculante(v):

    m = np.zeros((len(v), len(v)))

    for i in range(len(m)):
        for j in range(len(m[i])):
            pos_a_leer_de_v = (j - i) % len(v)
            m[i][j] = v[pos_a_leer_de_v]

    return m

# Ejercicio 13. Desarrollar una funcion matrizVandermonde(v), donde v es un
# vector de R^n y se devuelve la matriz de Vandermonde V en R^(n x n) cuya
# fila i-esima corresponde con la potencia (i-1)-esima de los elementos de v.
def matrizVandermonde(v):

    m = np.zeros((len(v), len(v)))

    for i in range(len(m)):
        for j in range(len(m[i])):
            m[i][j] = v[j] ** i

    return m

# Ejercicio 14. Desarrollar una funcion numeroAureo(n) que estime el numero
# aureo phi como F(k+1)/F(k), siendo F(k) el k-esimo numero de la sucesion
# de Fibonacci. Para esto, formulen la sucesion de Fibonacci
# F(k+1) = F(k) + F(k-1)
# de forma matricial, usando la semilla F0 = 0, F1 = 1.
def vector_fibonacci(k):
    v = np.array([1, 0])
    M = np.array([
        [1, 1],
        [1, 0]
    ])

    for i in range(k):
        v = calcularAx(M, v)

    return v

def fibonacci(k):
    v = vector_fibonacci(k)
    return v[1]

def numeroAureo(n):
    v = vector_fibonacci(n)
    return v[0] / v[1]

# Ejercicio 15. Desarrollar una funcion matrizFiboncacci(n), que genera una
# matriz A de n x n, y cada a_ij = F(i+j), siendo F(k) el k-esimo numero de
# la sucesion de Fibonacci (considerando F0 = 0, F1 = 1).
def matrizFiboncacci(n):

    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            A[i][j] = fibonacci(i + j)

    return A

# Ejercicio 16. Desarrollar una funcion matrizHilbert(n), que genera una
# matriz de Hilbert H de n x n, y cada h_ij = 1 / (i+j+1).
def matrizHilbert(n):

    A = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            A[i][j] = 1 / (i + j + 1)

    return A

# Ejercicio 17. Usando las funciones previamente desarrolladas donde sea
# posible, escriba una rutina que calcule los valores entre -1 y 1 de los
# siguientes polinomios:
#   - x^5 - x^4 + x^3 - x^2 + x - 1
#   - x^2 + 3
#   - x^10 - 2
# v[k] es el coeficiente de x^k. Devuelve el polinomio evaluado en 100 puntos
# equiespaciados entre -1 y 1.
def polinomios(v):
    n = 100
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        x[i] = -1 + i * (2 / 99)
    A = matrizVandermonde(x)  # O(n^2)
    for i in range(n):
        acc = 0
        for j in range(v.size):
            acc = acc + v[j] * A[j][i]
        y[i] = acc
    return y

# Ejercicio 18. Modificar la funcion row_echelon de manera que evalue en
# cada pivot si no hay otro elemento de la misma columna con modulo mayor
# (en valor absoluto). En caso afirmativo hacer el swap de las filas. Esta
# operatoria permite tener mayor estabilidad numerica.
# Parte de la version recursiva del notebook de L00: en vez de tomar como
# pivot el primer elemento no nulo de la columna, toma el de mayor modulo.
def row_echelon(M):
    A = np.copy(M)
    if issubclass(A.dtype.type, np.integer):
        A = A.astype(float)
    # Si A no tiene filas o columnas, ya esta escalonada
    f, c = A.shape
    if f == 0 or c == 0:
        return A
    # buscamos el elemento de mayor modulo de la primera columna
    i = 0
    for k in range(1, f):
        if abs(A[k, 0]) > abs(A[i, 0]):
            i = k
    if A[i, 0] == 0:
        # si todos los elementos de la primera columna son ceros
        # escalonamos filas desde la segunda columna
        B = row_echelon(A[:, 1:])
        return np.block([A[:, :1], B])
    # swap de filas para dejar el de mayor modulo como pivot
    if i > 0:
        intercambiarFilas(A, 0, i)
    # a las filas subsiguientes les restamos un multiplo de la primera
    for k in range(1, f):
        sumar_fila_multiplo(A, k, 0, -A[k, 0] / A[0, 0])
        A[k, 0] = 0.0  # evita que quede basura de redondeo debajo del pivot
    # escalonamos desde la segunda fila y segunda columna en adelante
    B = row_echelon(A[1:, 1:])
    return np.block([[A[:1, :]], [A[1:, :1], B]])
