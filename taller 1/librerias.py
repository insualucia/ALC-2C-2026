import numpy as np

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
    B = np.zeros((m,n))
    for i in range(len(A)):
        for j in range(A[i].size):
            B[j][i] = A[i][j]

    return B

# Ejercicio 7. Desarrollar una funcion esSimetrica(A) que devuelve True si la
# matriz A es simetrica y False en caso contrario.
def simetrica(A):

    if not(esCuadrada(A)):
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
        return None;

    b = np.zeros(len(A))
    for i in range(len(A)):
        sum = 0
        for j in range(A[i].size):
            sum += A[i][j] * x[j]
        b[i] = sum

    return b

# Ejercicio 9. Desarrollar una funcion intercambiarFilas(A, i, j), que
# intercambie las filas i y la j de la matriz A. El intercambio tiene que
# ser in-place.
def intercambiarFilas(A, i, j):
    # TODO: resolver
    pass

# Ejercicio 10. Desarrollar una funcion sumar_fila_multiplo(A, i, j, s) que
# a la fila i le sume la fila j multiplicada por un escalar s. Esta es una
# operacion elemental clave en la eliminacion gaussiana. La operacion debe
# ser in-place.
def sumar_fila_multiplo(A, i, j, s):
    # TODO: resolver
    pass

# Ejercicio 11. Desarrollar una funcion esDiagonalmenteDominante(A) que
# devuelva True si una matriz cuadrada A es estrictamente diagonalmente
# dominante. Esto ocurre si para cada fila, el valor absoluto del elemento
# en la diagonal es mayor que la suma de los valores absolutos de los demas
# elementos en esa fila.
def esDiagonalmenteDominante(A):
    # TODO: resolver
    pass

# Ejercicio 12. Desarrollar una funcion matrizCirculante(v) que genere una
# matriz circulante a partir de un vector. En una matriz circulante la
# primer fila es igual al vector v, y en cada fila se encuentra una
# permutacion ciclica de la fila anterior, moviendo los elementos un lugar
# hacia la derecha.
def matrizCirculante(v):
    # TODO: resolver
    pass

# Ejercicio 13. Desarrollar una funcion matrizVandermonde(v), donde v es un
# vector de R^n y se devuelve la matriz de Vandermonde V en R^(n x n) cuya
# fila i-esima corresponde con la potencia (i-1)-esima de los elementos de v.
def matrizVandermonde(v):
    # TODO: resolver
    pass

# Ejercicio 14. Desarrollar una funcion numeroAureo(n) que estime el numero
# aureo phi como F(k+1)/F(k), siendo F(k) el k-esimo numero de la sucesion
# de Fibonacci. Para esto, formulen la sucesion de Fibonacci
# F(k+1) = F(k) + F(k-1)
# de forma matricial, usando la semilla F0 = 0, F1 = 1. Grafique el valor
# aproximado de phi en funcion del numero de pasos de la sucesion
# considerado.
def numeroAureo(n):
    # TODO: resolver
    pass

# Ejercicio 15. Desarrollar una funcion matrizFiboncacci(n), que genera una
# matriz A de n x n, y cada a_ij = F(i+j), siendo F(k) el k-esimo numero de
# la sucesion de Fibonacci (considerando F0 = 0, F1 = 1).
def matrizFiboncacci(n):
    # TODO: resolver
    pass

# Ejercicio 16. Desarrollar una funcion matrizHilbert(n), que genera una
# matriz de Hilbert H de n x n, y cada h_ij = 1 / (i+j+1).
def matrizHilbert(n):
    # TODO: resolver
    pass

# Ejercicio 17. Usando las funciones previamente desarrolladas donde sea
# posible, escriba una rutina que calcule los valores entre -1 y 1 de los
# siguientes polinomios:
#   - x^5 - x^4 + x^3 - x^2 + x - 1
#   - x^2 + 3
#   - x^10 - 2
# Grafique el valor de los polinomios en el rango indicado, y calcule la
# cantidad de operaciones necesarias y el espacio en memoria para generar
# 100 puntos equiespaciados entre -1 y 1. Como crecen estos valores con n?
# Que modificaria para hacer el calculo mas eficiente?
def evaluarPolinomios():
    # TODO: resolver
    pass

# Ejercicio 18. Modificar la funcion row_echelon de manera que evalue en
# cada pivot si no hay otro elemento de la misma columna con modulo mayor
# (en valor absoluto). En caso afirmativo hacer el swap de las filas. Esta
# operatoria permite tener mayor estabilidad numerica.
def row_echelon(A):
    # TODO: resolver
    pass
