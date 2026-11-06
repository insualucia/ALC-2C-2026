"""
alc.py - Modulo ALC (Algebra Lineal Computacional, 2C 2026) + funciones del TP.

Archivo generado con tp/armar_alc.py a partir de los modulos del laboratorio (alc/)
y de tp/calibracion.py y tp/esquinas.py. No editar a mano: editar los originales
y volver a generarlo.
"""
import numpy as np


# ======================================================================
# L00 - Manipulacion de matrices + productos propios   (alc/basicos.py)
# ======================================================================

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


# ======================================================================
# L01 - Numeros de maquina   (alc/maquina.py)
# ======================================================================

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


# ======================================================================
# L02 - Transformaciones lineales   (alc/transformaciones.py)
# ======================================================================

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


# ======================================================================
# L03 - Normas y condicion   (alc/normas.py)
# ======================================================================

def norma(x, p):
    """
    Devuelve la norma p del vector x (p numérico o 'inf').
    """
    if p == 'inf':
        return float(np.max(np.abs(x)))
    return float(np.sum(np.abs(x) ** p) ** (1 / p))


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


# ======================================================================
# L04 - LU   (alc/lu.py)
# ======================================================================

def calculaLU(A):
    return elim_gaussiana(A)


def res_tri(L, b, inferior):
    n = L.shape[1]
    x = np.zeros(n)

    if inferior:
        for i in range(n):
            suma = 0
            for j in range(i):
                suma += L[i, j] * x[j]
            x[i] = (b[i] - suma) / L[i, i]

    else:
        for i in range(n-1, -1, -1):
            suma = 0
            for j in range(i+1, n):
                suma += L[i, j] * x[j]
            x[i] = (b[i] - suma) / L[i, i]
    return x

def inversa(A):
    n = A.shape[0]

    L, U, cant_op = elim_gaussiana(A)

    invA = np.zeros((n, n))

    for i in range(n):
        e = np.zeros(n)
        e[i] = 1
        y = res_tri(L, e, True)
        x = res_tri(U, y, False)
        invA[:, i] = x
    return invA



def calculaLDV (A):
    L, U, cant_op = elim_gaussiana(A)
    U_traspuesta = traspuesta(U)
    V_traspuesta, D, cant_op1 = elim_gaussiana(U_traspuesta)
    V = traspuesta(V_traspuesta)
    return L, D, V

def elim_gaussiana(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.astype(float)
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    ## desde aqui -- CODIGO A COMPLETAR
    for k in range(n-1):
        for i in range(k+1, n):
            multiplicador = Ac[i, k] / Ac[k, k]
            cant_op += 1
            Ac[i, k] = multiplicador
            for j in range(k + 1, n):
                Ac[i, j] = Ac[i,j] - multiplicador * Ac[k, j]
                cant_op += 2

    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i > j:
                L[i, j] = Ac[i, j]
            else:
                U[i, j] = Ac[i, j]

                
    ## hasta aqui, calculando L, U y la cantidad de operaciones sobre 
    ## la matriz Ac
            
    
    return L, U, cant_op

def esSDP(A, atol=1e-10):
    n = A.shape[1]
    for i in range(n):
        for j in range(n):
            if abs(A[i, j] - A[j, i]) > atol:
                return False
    L, D, V = calculaLDV(A)
    for i in range(n):
        if D[i, i] <= atol:
            return False
    return True
            


# ======================================================================
# L05 - QR   (alc/qr.py)
# ======================================================================

def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    """
    A una matriz de n x n
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A no es de n x n, debe retornar None
    """
    n = A.shape[0]
    m = A.shape[1]
    n_ops = 0
    if n != m:
        return None

    Q = np.zeros((n, m))
    R = np.zeros((n, m))

    for j in range(m):
        u = A[:, j]
        for i in range(j):
            R[i, j] = productoInterno(Q[:, i], A[:, j])
            n_ops += 2*n - 1  
            u = u - R[i, j] * Q[:, i]
            n_ops += 2*n 

        normaU = norma(u, 2)
        n_ops += 2*n 
        if normaU < tol:
            R[j, j] = 0.0
            Q[:, j] = np.zeros(n)
        else:
            R[j, j] = normaU
            Q[:, j] = u / normaU
            n_ops += n  

    if retorna_nops:
        return Q, R, n_ops
    return Q, R


def QR_con_HH(A, tol=1e-12, extras=False):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    extras : bool, opcional
        Si es True, devuelve informacion extra sobre el proceso de factorizacion.
        Por defecto es False. Esto lo hacemos para poder graficar el proceso.
    Devuelve la factorizacion QR de A usando reflectores de Householder.
    Devuelve:
        Q, R, extra_info (si extras es True)
        Q, R (si extras es False)
    extra_info es un diccionario con la clave:
        'R_matrices': lista de las matrices R en cada paso
        'Q_matrices': lista de las matrices Q en cada paso
    """
    # TODO: extras todavia no esta implementado
    m = A.shape[0]
    n = A.shape[1]
    if n > m:
        return None

    Q = np.eye(m)
    R = np.array(A, dtype=float)


    if extras:
        R_matrices = [R.copy()]
        Q_matrices = [Q.copy()]

    for j in range(n):
        columnaDesdeDiag = R[j:, j]
        norma_col = norma(columnaDesdeDiag, 2)

        if norma_col >= tol:
            signo = 1.0 if columnaDesdeDiag[0] >= 0 else -1.0
            alfa = -signo * norma_col

            v = columnaDesdeDiag.copy()
            v[0] -= alfa
            norma_v = norma(v, 2)

            if norma_v > tol:
                v = v / norma_v

                # v^T R  ==  R^T v
                vR = calcularAx(traspuesta(R[j:, j:]), v)
                R[j:, j:] -= 2.0 * productoExterno(v, vR)
                Qv = calcularAx(Q[:, j:], v)
                Q[:, j:] -= 2.0 * productoExterno(Qv, v)

        if extras:
            R_matrices.append(R.copy())
            Q_matrices.append(Q.copy())

    for i in range(m):
        for j in range(min(i, n)):
            if abs(R[i, j]) < tol:
                R[i, j] = 0.0

    if extras:
        R_matrices[-1] = R.copy()   # que coincida con el R final filtrado
        extra_info = {'R_matrices': R_matrices, 'Q_matrices': Q_matrices}
        return Q, R, extra_info
    else:
        return Q, R


def calculaQR(A, metodo='RH', tol=1e-12):
    """
    A una matriz de n x n
    tol la tolerancia con la que se filtran elementos nulos en R
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """
    if metodo == 'RH':
        return QR_con_HH(A, tol=tol)
    elif metodo == 'GS':
        return QR_con_GS(A, tol=tol)
    else:
        return None


# ======================================================================
# L06 - Autovalores   (alc/autovalores.py)
# ======================================================================

def transformacion_de_f(A, v):

    w = calcularAx(A, v)
    norm = norma(w, 2)
    if norm == 0:
        return 0
    return w / norm


def metpot2k(A, tol=1e-15, K=1000):
    """
    A una matriz de n x n
    tol la tolerancia en la diferencia entre un paso y el siguiente de la estimacion del autovector.
    K el numero maximo de iteraciones a realizarse.
    Retorna vector v, autovalor lambda y numero de iteracion realizadas k
    """
    v = np.zeros(A.shape[1])
    while np.all(v == 0):
        v = np.random.rand(A.shape[1])

    v_monio = transformacion_de_f(A, transformacion_de_f(A, v))
    e = productoInterno(v_monio, v)
    k = 0

    while np.abs(e - 1) > tol and k < K:
        v = v_monio
        v_monio = transformacion_de_f(A, transformacion_de_f(A, v))
        e = productoInterno(v_monio, v)
        k += 1

    l = productoInterno(v_monio, calcularAx(A, v_monio))

    return v_monio, l, k


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

    n = A.shape[0]
    if n == 1:
        return np.eye(1), np.array(A, dtype=float)

    v1, l1, _ = metpot2k(A, tol, K)

    # reflector de Householder que lleva v1 a e1
    w = -v1
    w[0] += 1
    norma_w = norma(w, 2)
    if norma_w < tol:
        # v1 ya es e1, no hace falta reflejar
        H = np.eye(n)
    else:
        H = np.eye(n) - 2 * productoExterno(w, w) / norma_w ** 2

    B = multiplicacionDeMatrices(multiplicacionDeMatrices(H, A), traspuesta(H))

    if n == 2:
        return H, B

    # B es simetrica en teoria, pero no exactamente por el redondeo: sin esto
    # la llamada recursiva devolveria None
    A_monio = B[1:, 1:]
    A_monio = (A_monio + traspuesta(A_monio)) / 2
    S_monio, D_monio = diagRH(A_monio, tol, K)

    D = np.zeros((n, n))
    D[0, 0] = l1
    D[1:, 1:] = D_monio

    S_bloque = np.zeros((n, n))
    S_bloque[0, 0] = 1
    S_bloque[1:, 1:] = S_monio
    S = multiplicacionDeMatrices(H, S_bloque)

    return S, D


# ======================================================================
# L07 - Markov y matrices ralas   (alc/markov.py)
# ======================================================================

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


# ======================================================================
# L08 - SVD   (alc/svd.py)
# ======================================================================

def svd_reducida(A, k="max", tol=1e-15):
    """
    A la matriz de interes (de m x n)
    k es el numero de valores singulares (y vectores) a retener.
    tol es la tolerancia para considerar un valor singular igual a cero.
    Retorna hatU (m x k), hatSig (vector de k valores singulares) y hatV (n x k).
    """
    A = np.asarray(A, dtype=float)
    m = A.shape[0]
    n = A.shape[1]

    if m < n:                       
        U, S, V = svd_reducida(A.T, k, tol)
        return V, S, U

    S_aut, D = diagRH(multiplicacionDeMatrices(traspuesta(A), A))

    lam = np.zeros(n)
    for i in range(n):
        lam[i] = D[i, i]
    V = np.array(S_aut)

    # ordeno de mayor a menor, permutando columnas de V
    for i in range(n):
        imax = i
        for j in range(i + 1, n):
            if lam[j] > lam[imax]:
                imax = j
        if imax != i:
            aux = lam[i]; lam[i] = lam[imax]; lam[imax] = aux
            col = np.array(V[:, i]); V[:, i] = V[:, imax]; V[:, imax] = col
    S = np.zeros(n)
    for i in range(n):
        if lam[i] > 0:
            S[i] = np.sqrt(lam[i])

    # cuantos retener: sigma > tol y lambda no es solo ruido de redondeo
    if k == "max":
        r = 0
        for i in range(n):
            if S[i] > tol and lam[i] > 1e-12 * lam[0]:
                r += 1
    else:
        # k explicito: se retienen los k pedidos aunque sean chicos (los
        # que sean exactamente cero se descartan para no dividir por 0)
        r = 0
        for i in range(min(int(k), n)):
            if S[i] > tol:
                r += 1

    # u_i = A v_i / sigma_i
    AV = multiplicacionDeMatrices(A, V[:, :r])
    hatU = np.zeros((m, r))
    for i in range(r):
        hatU[:, i] = AV[:, i] / S[i]

    return hatU, S[:r], V[:, :r]


# ======================================================================
# TP - Calibracion de la camara (Ej. 2 y 3)   (tp/calibracion.py)
# ======================================================================

def obtenerQ3d(imgPts, objPts):
    """
    Arma la matriz Q (2n x 12) del sistema Qm = 0 (ecuacion 6).
    imgPts: array de n x 2 con los (u, v) de cada punto en la imagen.
    objPts: array de n x 3 con los (U, V, W) de cada punto en el mundo.
    Retorna Q, donde cada punto aporta dos filas:
        [P^T, 0, -u P^T]  y  [0, P^T, -v P^T],  con P = (U, V, W, 1).
    """
    imgPts = np.asarray(imgPts, dtype=float)
    objPts = np.asarray(objPts, dtype=float)
    n = objPts.shape[0]
    Q = np.zeros((2 * n, 12))
    for i in range(n):
        u = imgPts[i, 0]
        v = imgPts[i, 1]
        P = np.array([objPts[i, 0], objPts[i, 1], objPts[i, 2], 1.0])
        Q[2 * i, 0:4] = P
        Q[2 * i, 8:12] = -u * P
        Q[2 * i + 1, 4:8] = P
        Q[2 * i + 1, 8:12] = -v * P
    return Q


def resolverDLT3D(Q):
    """
    Resuelve  min ||Qm||^2  sujeto a ||m|| = 1  (ecuacion 8).
    La solucion es el autovector de Q^T Q de menor autovalor (el vector singular
    derecho de Q asociado al menor valor singular).
    Q: matriz de 2n x 12.
    Retorna M, la matriz de proyeccion de 3 x 4 (m reordenado por filas).
    """
    Q = np.asarray(Q, dtype=float)
    # autovector de menor autovalor de Q^T Q (ver _vectorNucleo): no se usa
    # svd_reducida porque descarta los valores singulares casi nulos
    m = _vectorNucleo(Q)
    M = np.zeros((3, 4))
    for j in range(3):
        M[j, :] = m[4 * j:4 * j + 4]
    return M


def _productoTriple(r1, r2, r3):
    """Determinante de la matriz con columnas r1, r2, r3: r1 . (r2 x r3)."""
    c = np.array([r2[1] * r3[2] - r2[2] * r3[1],
                  r2[2] * r3[0] - r2[0] * r3[2],
                  r2[0] * r3[1] - r2[1] * r3[0]])
    return np.sum(r1 * c)


def _norma(v):
    """Norma euclidea de un vector."""
    return np.sqrt(np.sum(v * v))


def devolverMatricesCalibracion3d(M, imponerAlfasIguales=False, imponerGammaCero=False):

    """
    Descompone la matriz de proyección M = K [R t] en sus parámetros
    intrínsecos (K) y extrínsecos (R, t), imponiendo que R sea ortonormal.

    Parámetros:
        M : array (3, 4), matriz de proyección (a menos de escala), por
            ejemplo la que devuelve resolverDLT3D.
        imponerAlfasIguales : bool, opcional. Si es True, fuerza alfa_u = alfa_v
            (píxeles cuadrados), reemplazando ambos por su promedio.
        imponerGammaCero : bool, opcional. Si es True, fuerza K12 = gamma = 0
            (ejes del sensor perpendiculares).

    Devuelve:
        R : array (3, 3), matriz de rotación (ortonormal).
        t : array (3,), vector de traslación, en las unidades del mundo.
        K : array (3, 3), matriz de parámetros intrínsecos, con K33 = 1.
    """

    M = np.array(M, dtype=float)

    # --- 1. Sacar la escala desconocida: dividir M por la norma de m3 ---
    # Las tres primeras entradas de la fila 3 son c * r3, y r3 tiene norma 1.
    M = M / norma(M[2, 0:3], 2)

    # --- 2. Elegir el signo: la mira tiene que estar delante de la cámara ---
    # t3 = M[2, 3] tiene que ser positivo; si no, cambiamos el signo de M.
    if M[2, 3] < 0:
        M = -M

    # Bloque izquierdo (K R): m1, m2, m3 son las tres primeras entradas de cada fila
    m1 = M[0, 0:3]
    m2 = M[1, 0:3]
    m3 = M[2, 0:3]

    # --- 3. Despeje de abajo hacia arriba usando la ortonormalidad de R ---
    # Fila 3: m3 = r3
    r3 = m3

    # Fila 2: m2 = alfa_v r2 + v0 r3
    v0 = productoInterno(m2, r3)                    # r3 . m2 = v0
    alfaV = norma(m2 - v0 * r3, 2)                  # ||m2 - v0 r3|| = alfa_v
    r2 = (m2 - v0 * r3) / alfaV

    # Fila 1: m1 = alfa_u r1 + gamma r2 + u0 r3
    u0 = productoInterno(m1, r3)                    # r3 . m1 = u0
    gamma = productoInterno(m1, r2)                 # r2 . m1 = gamma
    alfaU = norma(m1 - gamma * r2 - u0 * r3, 2)     # ||m1 - gamma r2 - u0 r3|| = alfa_u
    r1 = (m1 - gamma * r2 - u0 * r3) / alfaU

    # Armamos R con r1, r2, r3 como filas
    R = np.vstack([r1, r2, r3])

    # --- 4. Restricciones opcionales sobre K (Ejercicio 2d) ---
    if imponerGammaCero:
        gamma = 0.0
    if imponerAlfasIguales:
        alfaPromedio = (alfaU + alfaV) / 2
        alfaU = alfaPromedio
        alfaV = alfaPromedio

    K = np.array([[alfaU, gamma, u0],
                  [0.0,   alfaV, v0],
                  [0.0,   0.0,   1.0]])

    # --- 5. Traslación: la última columna de M es K t (sistema triangular) ---
    # Se resuelve por sustitución hacia atrás, con la K ya definida.
    m14 = M[0, 3]
    m24 = M[1, 3]
    m34 = M[2, 3]
    t3 = m34
    t2 = (m24 - v0 * t3) / alfaV
    t1 = (m14 - gamma * t2 - u0 * t3) / alfaU
    t = np.array([t1, t2, t3])

    return R, t, K

def obtenerQ2d(imgPts, objPts):
    """
    Arma la matriz Q (2n x 9) del sistema Qh = 0 (ecuacion 11), para puntos coplanares.
    imgPts: array de n x 2 con los (u, v) de cada punto en la imagen.
    objPts: array de n x 2 con los (U, V) de cada punto en el plano del objeto (W = 0).
            Si viene de n x 3, la tercera columna debe ser 0 y se ignora.
    Retorna Q, donde cada punto aporta dos filas:
        [P^T, 0, -u P^T]  y  [0, P^T, -v P^T],  con P = (U, V, 1).
    """
    imgPts = np.asarray(imgPts, dtype=float)
    objPts = np.asarray(objPts, dtype=float)

    # Si vienen coordenadas 3D, verificar que sean coplanares en W = 0
    if objPts.shape[1] == 3 and np.any(np.abs(objPts[:, 2]) > 0):
        raise ValueError("obtenerQ2d: los puntos deben tener W = 0 (plano del objeto).")

    n = objPts.shape[0]
    Q = np.zeros((2 * n, 9))
    for i in range(n):
        u = imgPts[i, 0]
        v = imgPts[i, 1]
        # Solo U y V: la columna de W desaparece porque W = 0
        P = np.array([objPts[i, 0], objPts[i, 1], 1.0])
        # Fila de u: h1^T P - u * h3^T P = 0
        Q[2 * i, 0:3] = P
        Q[2 * i, 6:9] = -u * P
        # Fila de v: h2^T P - v * h3^T P = 0
        Q[2 * i + 1, 3:6] = P
        Q[2 * i + 1, 6:9] = -v * P
    return Q



def verificarCalibracion3d(M, R, t, K, imgPts, objPts, tol=1e-8):
    """
    Chequeos de la calibracion obtenida con devolverMatricesCalibracion3d.
    M: matriz de proyeccion (3 x 4); R, t, K: resultado de la descomposicion.
    imgPts (n x 2) y objPts (n x 3): puntos usados para calibrar.
    Retorna un diccionario con:
      ortogonalidad: max |R R^T - I|          (debe ser ~ 0)
      det_R:         determinante de R        (debe ser +1)
      K_triangular:  max |K bajo la diagonal| (debe ser ~ 0), K33 = 1
      dif_M:         max |K [R|t] - M/escala| (debe ser ~ 0)
      error_reproyeccion: error maximo en pixeles al reproyectar objPts
      delante_camara: True si todos los puntos tienen Z > 0 en la camara
    """
    imgPts = np.asarray(imgPts, dtype=float)
    objPts = np.asarray(objPts, dtype=float)
    n = objPts.shape[0]

    RRt = multiplicacionDeMatrices(R, traspuesta(R))
    ortogonalidad = np.max(np.abs(RRt - np.eye(3)))
    det_R = _productoTriple(R[:, 0], R[:, 1], R[:, 2])
    K_triangular = max(abs(K[1, 0]), abs(K[2, 0]), abs(K[2, 1]))

    # K [R | t] debe coincidir con M salvo un factor de escala (global)
    Rt = np.hstack([R, t.reshape(3, 1)])
    KRt = multiplicacionDeMatrices(K, Rt)
    escala = np.sum(M * KRt) / np.sum(KRt * KRt)
    dif_M = np.max(np.abs(M - escala * KRt))

    # reproyeccion y profundidad en el sistema de la camara
    P = np.hstack([objPts, np.ones((n, 1))])
    proy = multiplicacionDeMatrices(P, traspuesta(multiplicacionDeMatrices(K, Rt)))
    uv = proy[:, 0:2] / proy[:, 2:3]
    error_reproyeccion = np.max(np.abs(uv - imgPts))
    Zcam = multiplicacionDeMatrices(P, traspuesta(Rt))[:, 2]
    delante_camara = bool(np.all(Zcam > 0))

    return {"ortogonalidad": ortogonalidad, "det_R": det_R,
            "K_triangular": K_triangular, "dif_M": dif_M,
            "error_reproyeccion": error_reproyeccion,
            "delante_camara": delante_camara}



def _vectorNucleo(M):
    """
    Vector unitario x que minimiza ||M x|| (a menos de signo): el autovector de
    M^T M de menor autovalor, calculado con diagRH del modulo ALC.
    Se usa en lugar de svd_reducida porque esta descarta los valores singulares
    (casi) nulos, y con datos exactos el menor valor singular es justo ~0: la
    ultima columna de V ya no seria el vector del nucleo.
    M: matriz de m x n. Retorna un vector de n componentes.
    """
    M = np.asarray(M, dtype=float)
    S, D = diagRH(multiplicacionDeMatrices(traspuesta(M), M))
    n = M.shape[1]
    imin = 0
    for j in range(n):
        if D[j, j] < D[imin, imin]:
            imin = j
    return np.array(S)[:, imin]


def _normalizacionIsotropica(pts):
    """
    Normalizacion de Hartley (seccion 2.3): matriz T (3 x 3) tal que los puntos
    T [x y 1]^T tienen centroide en el origen y distancia media al origen sqrt(2).
    pts: array de n x 2. Retorna T.
    """
    pts = np.asarray(pts, dtype=float)
    n = pts.shape[0]
    cx = np.sum(pts[:, 0]) / n
    cy = np.sum(pts[:, 1]) / n
    distMedia = np.sum(np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)) / n
    s = np.sqrt(2) / distMedia
    return np.array([[s, 0.0, -s * cx],
                     [0.0, s, -s * cy],
                     [0.0, 0.0, 1.0]])


def _aplicarT(T, pts):
    """Aplica la transformacion homogenea T (3 x 3) a puntos 2D (n x 2)."""
    pts = np.asarray(pts, dtype=float)
    n = pts.shape[0]
    P = np.hstack([pts, np.ones((n, 1))])
    Y = multiplicacionDeMatrices(P, traspuesta(T))
    return Y[:, 0:2] / Y[:, 2:3]


def resolverH(imgPts, objPts, normalizar=True):
    """
    Estima la homografia H (3 x 3) del plano del tablero a la imagen, con
    lambda (u, v, 1)^T = H (U, V, 1)^T, resolviendo  min ||Qh||^2 s.a. ||h|| = 1.
    imgPts: n x 2 (u, v) en pixeles. objPts: n x 2 (U, V) en el plano (o n x 3 con W = 0).
    normalizar: si es True usa la normalizacion isotropica de Hartley (ec. 14):
        se resuelve para los puntos normalizados y luego H = T_img^-1 H_norm T_world.
    Retorna H (3 x 3) con norma de Frobenius 1.
    """
    imgPts = np.asarray(imgPts, dtype=float)
    objPts = np.asarray(objPts, dtype=float)[:, 0:2]
    if normalizar:
        Timg = _normalizacionIsotropica(imgPts)
        Tworld = _normalizacionIsotropica(objPts)
        imgN = _aplicarT(Timg, imgPts)
        objN = _aplicarT(Tworld, objPts)
    else:
        imgN = imgPts
        objN = objPts
    Q = obtenerQ2d(imgN, objN)
    h = _vectorNucleo(Q)
    H = np.zeros((3, 3))
    for j in range(3):
        H[j, :] = h[3 * j:3 * j + 3]
    if normalizar:
        H = multiplicacionDeMatrices(multiplicacionDeMatrices(inversa(Timg), H), Tworld)
    return H / np.sqrt(np.sum(H * H))


def _vij(H, i, j):
    """
    Vector v_ij (6,) con  c_i^T B c_j = v_ij^T b,  donde c_i es la columna i de H
    y b = (B11, B12, B22, B13, B23, B33). (Zhang, seccion 3.)
    """
    hi = H[:, i]
    hj = H[:, j]
    return np.array([hi[0] * hj[0],
                     hi[0] * hj[1] + hi[1] * hj[0],
                     hi[1] * hj[1],
                     hi[2] * hj[0] + hi[0] * hj[2],
                     hi[2] * hj[1] + hi[1] * hj[2],
                     hi[2] * hj[2]])


def armarA(Hlst):
    """
    Arma la matriz A (2N x 6) del sistema A b = 0 (ecuacion 13), con N homografias.
    Hlst: lista de N matrices H_j de 3 x 3 (una por vista).
    Cada vista aporta dos filas:  v12^T b = 0  y  (v11 - v22)^T b = 0,
    que salen de r1^T r2 = 0 y ||r1|| = ||r2||. Se normaliza cada H a norma 1
    para que todas las vistas pesen parecido.
    """
    N = len(Hlst)
    A = np.zeros((2 * N, 6))
    for k in range(N):
        H = np.asarray(Hlst[k], dtype=float)
        H = H / np.sqrt(np.sum(H * H))
        A[2 * k, :] = _vij(H, 0, 1)
        A[2 * k + 1, :] = _vij(H, 0, 0) - _vij(H, 1, 1)
    return A


def _cholesky(B):
    """
    Factorizacion de Cholesky B = L L^T (L triangular inferior) de una matriz
    simetrica definida positiva, a partir de la factorizacion LDV del modulo ALC:
    para B simetrica B = L1 D L1^T, y si D > 0, L = L1 D^(1/2).
    Lanza ValueError si algun elemento de D es <= 0 (B no es definida positiva).
    """
    n = B.shape[0]
    L1, D, V = calculaLDV(B)
    L = np.zeros((n, n))
    for j in range(n):
        if D[j, j] <= 0:
            raise ValueError("_cholesky: la matriz no es definida positiva.")
        L[:, j] = L1[:, j] * np.sqrt(D[j, j])
    return L


def calcularMatrizK2D(A):
    """
    Calcula la matriz de intrinsecos K (3 x 3) a partir de A (2N x 6).
    1) b = vector singular derecho de menor valor singular de A (A b = 0, ||b|| = 1).
    2) b es definido salvo signo: se elige el que deja a B = K^-T K^-1 definida
       positiva (B11 > 0 y Cholesky exitoso).
    3) B = L L^T (Cholesky) => L = K^-T => K = (L^T)^-1, normalizada con K33 = 1.
    Se necesitan N >= 3 vistas (N >= 2 si se supone gamma = 0), con poses no degeneradas.
    """
    A = np.asarray(A, dtype=float)
    b = _vectorNucleo(A)
    if b[0] < 0:
        b = -b
    B = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]],
                  [b[3], b[4], b[5]]])
    L = _cholesky(B)
    K = inversa(traspuesta(L))
    return K / K[2, 2]


def obtenerPose2D(Hj, K):
    """
    Pose (R_j, t_j) de la vista j a partir de su homografia H_j y de K (ec. 12 y h).
    H_j ~ K [r1 r2 t]  =>  r1 = lam K^-1 c1,  r2 = lam K^-1 c2,  r3 = r1 x r2,
    t = lam K^-1 c3,  con lam = 1 / ||K^-1 c1||.
    El signo de lam se elige para que t3 > 0 (el tablero queda delante de la camara).
    Como el ruido hace que [r1 r2 r3] no sea exactamente una rotacion, se la
    proyecta a la rotacion ortogonal mas cercana (R = U V^T de su SVD), calculada
    con iteracion de Newton.
    Retorna R (3 x 3) y t (3,) en las unidades del tablero (mm).
    """
    Hj = np.asarray(Hj, dtype=float)
    Kinv = inversa(K)
    c1 = Hj[:, 0]
    c2 = Hj[:, 1]
    c3 = Hj[:, 2]
    k1 = calcularAx(Kinv, c1)
    k2 = calcularAx(Kinv, c2)
    k3 = calcularAx(Kinv, c3)
    lam = 1.0 / norma(k1, 2)
    if lam * k3[2] < 0:
        lam = -lam
    r1 = lam * k1
    r2 = lam * k2
    r3 = np.array([r1[1] * r2[2] - r1[2] * r2[1],
                   r1[2] * r2[0] - r1[0] * r2[2],
                   r1[0] * r2[1] - r1[1] * r2[0]])
    t = lam * k3
    R = np.array([r1, r2, r3]).T          # r1, r2, r3 como columnas
    # Proyeccion a la rotacion mas cercana (el factor U V^T de la SVD de R) por
    # iteracion de Newton R <- (R + R^-T) / 2: converge en 2-3 pasos a la misma
    # matriz y queda ortonormal a precision de maquina. (Con svd_reducida de ALC
    # quedaba ortonormal solo hasta ~3e-4: R ya es casi rotacion, asi que R^T R ~ I
    # tiene los 3 autovalores casi iguales y sus autovectores salen mal separados.)
    for _ in range(50):
        Rnueva = (R + traspuesta(inversa(R))) / 2
        cambio = np.max(np.abs(Rnueva - R))
        R = Rnueva
        if cambio < 1e-15:
            break
    return R, t


# ======================================================================
# TP - Deteccion de esquinas del tablero (Ej. 3)   (tp/esquinas.py)
# ======================================================================

def _suavizar(I, veces=2, r=2):
    """Suaviza la imagen I (2D) con un filtro de caja de lado 2r+1, aplicado
    `veces` veces (aproxima una gaussiana). Se hace separable, por filas y columnas."""
    I = np.array(I, dtype=float)
    for _ in range(veces):
        A = np.zeros(I.shape)
        for d in range(-r, r + 1):
            A[:, max(0, -d):I.shape[1] - max(0, d)] += I[:, max(0, d):I.shape[1] - max(0, -d)]
        A = A / (2 * r + 1)
        B = np.zeros(I.shape)
        for d in range(-r, r + 1):
            B[max(0, -d):I.shape[0] - max(0, d), :] += A[max(0, d):I.shape[0] - max(0, -d), :]
        I = B / (2 * r + 1)
    return I


def respuestaEsquinas(imagen):
    """
    Respuesta de 'silla de montar' en cada pixel: det(Hessiana) de la imagen
    suavizada, con signo cambiado. Las esquinas internas de un tablero (donde se
    tocan dos cuadrados negros en diagonal) son puntos silla: det(H) muy negativo,
    es decir, respuesta muy positiva.
    imagen: array (alto, ancho, 3) o (alto, ancho) con valores en [0, 1].
    Retorna un array (alto, ancho) con la respuesta (0 en el borde).
    """
    I = np.array(imagen, dtype=float)
    if I.ndim == 3:
        I = (I[:, :, 0] + I[:, :, 1] + I[:, :, 2]) / 3
    I = _suavizar(I)
    Ixx = np.zeros(I.shape)
    Iyy = np.zeros(I.shape)
    Ixy = np.zeros(I.shape)
    Ixx[:, 1:-1] = I[:, 2:] - 2 * I[:, 1:-1] + I[:, :-2]
    Iyy[1:-1, :] = I[2:, :] - 2 * I[1:-1, :] + I[:-2, :]
    Ixy[1:-1, 1:-1] = (I[2:, 2:] - I[2:, :-2] - I[:-2, 2:] + I[:-2, :-2]) / 4
    return -(Ixx * Iyy - Ixy * Ixy)


def candidatosEsquinas(R, cantidad=120, radio=12, umbral=0.15):
    """
    Maximos locales de la respuesta R, con supresion de no-maximos.
    Se toma repetidamente el maximo, se lo refina con el centroide ponderado de
    R en un entorno de 3 pixeles (precision subpixel) y se anula R a su alrededor.
    Retorna array (k, 2) de (u, v) con k <= cantidad.
    """
    R = np.array(R, dtype=float)
    # el borde de la imagen da falsas respuestas: se anula
    R[:10, :] = 0; R[-10:, :] = 0; R[:, :10] = 0; R[:, -10:] = 0
    tope = np.max(R)
    pts = []
    for _ in range(cantidad):
        m = np.max(R)
        if m < umbral * tope:
            break
        # posicion del maximo: primero la fila (maximo por filas), luego la columna
        maxFilas = np.max(R, axis=1)
        v = 0
        while maxFilas[v] != m:
            v += 1
        u = 0
        while R[v, u] != m:
            u += 1
        v0 = max(v - 3, 0); v1 = min(v + 4, R.shape[0])
        u0 = max(u - 3, 0); u1 = min(u + 4, R.shape[1])
        W = R[v0:v1, u0:u1]
        W = W * (W > 0)
        gu = np.arange(u0, u1) + 0.0
        gv = np.arange(v0, v1) + 0.0
        sw = np.sum(W)
        pts.append([np.sum(W * gu.reshape(1, -1)) / sw, np.sum(W * gv.reshape(-1, 1)) / sw])
        R[max(v - radio, 0):v + radio + 1, max(u - radio, 0):u + radio + 1] = 0
    return np.array(pts)


def _masCercano(pts, x, excluir=None):
    """Indice del punto de pts (k x 2) mas cercano a x, y la distancia."""
    d = np.sqrt(np.sum((pts - x.reshape(1, 2)) ** 2, axis=1))
    if excluir is not None:
        d = d + 1e9 * excluir
    i = 0
    for k in range(len(d)):
        if d[k] < d[i]:
            i = k
    return i, d[i]


def ordenarTablero(pts, nu=11, nv=7):
    """
    Dado un conjunto de candidatos a esquina (que incluye tambien esquinas
    externas del tablero), encuentra el bloque de nu x nv esquinas internas y las
    devuelve ordenadas: indice k = j * nu + i, con i = 0..nu-1 hacia la derecha
    (eje U) y j = 0..nv-1 hacia abajo (eje V).
    Idea: se etiqueta cada candidato con coordenadas enteras (i, j) de la grilla
    creciendo desde el punto central (siempre interno), prediciendo cada vecino
    con el paso local; luego se busca la ventana nu x nv completamente ocupada.
    Retorna array (nu*nv, 2), o None si no se pudo ordenar.
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    centro = np.array([np.sum(pts[:, 0]) / n, np.sum(pts[:, 1]) / n])
    sem, _ = _masCercano(pts, centro)
    # paso de la grilla: distancia al vecino mas cercano del semillero
    usados = np.zeros(n)
    usados[sem] = 1
    v1, paso = _masCercano(pts, pts[sem], usados)
    # los 4 vecinos de la grilla: puntos a distancia ~paso del semillero
    vec = []
    for k in range(n):
        d = np.sqrt(np.sum((pts[k] - pts[sem]) ** 2))
        if k != sem and d < 1.3 * paso:
            vec.append(pts[k] - pts[sem])
    # direccion "derecha": mayor componente x con |y| chico; "abajo" analogo
    a1 = None
    a2 = None
    for w in vec:
        if abs(w[0]) > abs(w[1]) and w[0] > 0:
            a1 = w
        if abs(w[1]) > abs(w[0]) and w[1] > 0:
            a2 = w
    if a1 is None or a2 is None:
        return None
    # crecimiento: etiquetas[(i, j)] = indice de punto
    etiq = {(0, 0): sem}
    pos = {(0, 0): pts[sem]}
    cola = [(0, 0)]
    while len(cola) > 0:
        i, j = cola.pop(0)
        for di, dj, base in [(1, 0, a1), (-1, 0, -a1), (0, 1, a2), (0, -1, -a2)]:
            q = (i + di, j + dj)
            if q in etiq:
                continue
            # paso local: si el vecino opuesto ya esta, se usa su diferencia
            prev = (i - di, j - dj)
            if prev in pos:
                salto = pos[(i, j)] - pos[prev]
            else:
                salto = base
            pred = pos[(i, j)] + salto
            k, d = _masCercano(pts, pred)
            ya = False
            for kk in etiq.values():
                if kk == k:
                    ya = True
            if d < 0.3 * paso and not ya:
                etiq[q] = k
                pos[q] = pts[k]
                cola.append(q)
    # rango de etiquetas: incluye el anillo exterior, asi que el bloque interno
    # es la ventana nu x nv completamente ocupada y CENTRADA en ese rango
    imin = 10 ** 6; imax = -10 ** 6; jmin = 10 ** 6; jmax = -10 ** 6
    for (i, j) in etiq:
        imin = min(imin, i); imax = max(imax, i)
        jmin = min(jmin, j); jmax = max(jmax, j)
    mejor = None
    mejorScore = 10 ** 9
    for i0 in range(imin, imax + 1):
        for j0 in range(jmin, jmax + 1):
            ok = True
            for j in range(nv):
                for i in range(nu):
                    if (i0 + i, j0 + j) not in etiq:
                        ok = False
            if ok:
                score = abs(2 * i0 + nu - 1 - imin - imax) + abs(2 * j0 + nv - 1 - jmin - jmax)
                if score < mejorScore:
                    mejorScore = score
                    mejor = (i0, j0)
    if mejor is None:
        return None
    i0, j0 = mejor
    out = np.zeros((nu * nv, 2))
    for j in range(nv):
        for i in range(nu):
            out[j * nu + i] = pts[etiq[(i0 + i, j0 + j)]]
    return out


def esquinasTablero(imagen, nu=11, nv=7):
    """Esquinas internas (nu*nv x 2) de una imagen de tablero, ordenadas, o None."""
    return ordenarTablero(candidatosEsquinas(respuestaEsquinas(imagen)), nu, nv)
