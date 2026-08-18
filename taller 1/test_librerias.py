"""
Tests (pytest) para librerias.py, uno (o mas) por cada ejercicio del
Laboratorio N1 (L00-Ejercicios.pdf).

Los tests reflejan lo que pide la CONSIGNA de cada ejercicio, no la
implementacion actual de librerias.py. Por lo tanto:
  - Los ejercicios 9 a 18 todavia no estan resueltos (son "pass" en el
    archivo), asi que sus tests van a fallar/errorar hasta que se
    implementen.
  - Algunos de los ejercicios 1 a 8 ya implementados tienen bugs (ver
    revision hecha en la conversacion), por lo que tambien pueden fallar
    con la version actual del codigo. Eso es esperado: el test marca
    cual es el comportamiento correcto.

Correr con:
    pytest "taller 1/test_librerias.py" -v
"""

import numpy as np
import pytest

from librerias import (
    esCuadrada,
    triangSup,
    triangInf,
    diagonal,
    traza,
    traspuesta,
    simetrica,
    calcularAx,
    intercambiarFilas,
    sumar_fila_multiplo,
    esDiagonalmenteDominante,
    matrizCirculante,
    matrizVandermonde,
    numeroAureo,
    matrizFiboncacci,
    matrizHilbert,
    evaluarPolinomios,
    row_echelon,
)


def fibonacci(k):
    """Sucesion de Fibonacci auxiliar para los tests (F0=0, F1=1)."""
    a, b = 0, 1
    for _ in range(k):
        a, b = b, a + b
    return a


# ---------------------------------------------------------------------
# Ejercicio 1: esCuadrada(A) -> True si A es cuadrada, False si no.
# ---------------------------------------------------------------------

def test_ej01_esCuadrada_matriz_cuadrada():
    A = np.array([[1, 2], [3, 4]])
    assert esCuadrada(A) is True


def test_ej01_esCuadrada_matriz_no_cuadrada():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    assert esCuadrada(A) is False


# ---------------------------------------------------------------------
# Ejercicio 2: triangSup(A) -> matriz U triangular superior de A, sin
# su diagonal.
# ---------------------------------------------------------------------

def test_ej02_triangSup():
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    esperado = np.array([[0, 2, 3],
                         [0, 0, 6],
                         [0, 0, 0]])
    np.testing.assert_array_equal(triangSup(A.copy()), esperado)


# ---------------------------------------------------------------------
# Ejercicio 3: triangInf(A) -> matriz L triangular inferior de A, sin
# su diagonal.
# ---------------------------------------------------------------------

def test_ej03_triangInf():
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    esperado = np.array([[0, 0, 0],
                         [4, 0, 0],
                         [7, 8, 0]])
    np.testing.assert_array_equal(triangInf(A.copy()), esperado)


# ---------------------------------------------------------------------
# Ejercicio 4: diagonal(A) -> matriz D diagonal de A.
# ---------------------------------------------------------------------

def test_ej04_diagonal():
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    esperado = np.array([[1, 0, 0],
                         [0, 5, 0],
                         [0, 0, 9]])
    np.testing.assert_array_equal(diagonal(A.copy()), esperado)


# ---------------------------------------------------------------------
# Ejercicio 5: traza(A) -> suma de los elementos de la diagonal de A.
# ---------------------------------------------------------------------

def test_ej05_traza():
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    assert traza(A.copy()) == 15


# ---------------------------------------------------------------------
# Ejercicio 6: traspuesta(A) -> matriz traspuesta de A (incluye caso
# no cuadrado, n x m -> m x n).
# ---------------------------------------------------------------------

def test_ej06_traspuesta_cuadrada():
    A = np.array([[1, 2], [3, 4]])
    esperado = np.array([[1, 3], [2, 4]])
    np.testing.assert_array_equal(traspuesta(A.copy()), esperado)


def test_ej06_traspuesta_no_cuadrada():
    A = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
    esperado = np.array([[1, 4], [2, 5], [3, 6]])  # 3x2
    np.testing.assert_array_equal(traspuesta(A.copy()), esperado)


# ---------------------------------------------------------------------
# Ejercicio 7: esSimetrica(A) -> True si A es simetrica, False si no.
# (en librerias.py la funcion se llama "simetrica")
# ---------------------------------------------------------------------

def test_ej07_simetrica_matriz_simetrica():
    A = np.array([[1, 2, 3],
                  [2, 5, 6],
                  [3, 6, 9]])
    assert simetrica(A.copy()) is True


def test_ej07_simetrica_matriz_no_simetrica():
    A = np.array([[1, 2], [3, 4]])
    assert simetrica(A.copy()) is False


def test_ej07_simetrica_matriz_no_cuadrada():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    assert simetrica(A.copy()) is False


# ---------------------------------------------------------------------
# Ejercicio 8: calcularAx(A, x) -> A (n x m) . x (largo m) = b (largo n)
# ---------------------------------------------------------------------

def test_ej08_calcularAx_cuadrada():
    A = np.array([[1, 2], [3, 4]])
    x = np.array([1, 1])
    esperado = np.array([3, 7])
    np.testing.assert_array_equal(calcularAx(A.copy(), x.copy()), esperado)


def test_ej08_calcularAx_no_cuadrada():
    A = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
    x = np.array([1, 0, 1])  # largo 3
    esperado = np.array([4, 10])  # largo 2
    np.testing.assert_array_equal(calcularAx(A.copy(), x.copy()), esperado)


# ---------------------------------------------------------------------
# Ejercicio 9: intercambiarFilas(A, i, j) -> intercambia in-place las
# filas i y j de A.
# ---------------------------------------------------------------------

def test_ej09_intercambiarFilas():
    A = np.array([[1, 2], [3, 4], [5, 6]])
    intercambiarFilas(A, 0, 2)
    esperado = np.array([[5, 6], [3, 4], [1, 2]])
    np.testing.assert_array_equal(A, esperado)


def test_ej09_intercambiarFilas_es_inplace():
    A = np.array([[1, 2], [3, 4]])
    id_original = id(A)
    intercambiarFilas(A, 0, 1)
    assert id(A) == id_original


# ---------------------------------------------------------------------
# Ejercicio 10: sumar_fila_multiplo(A, i, j, s) -> in-place, fila i +=
# s * fila j.
# ---------------------------------------------------------------------

def test_ej10_sumar_fila_multiplo():
    A = np.array([[1, 2], [3, 4]])
    sumar_fila_multiplo(A, 0, 1, 2)
    # fila0 = fila0 + 2*fila1 = [1,2] + 2*[3,4] = [7,10]
    esperado = np.array([[7, 10], [3, 4]])
    np.testing.assert_array_equal(A, esperado)


def test_ej10_sumar_fila_multiplo_es_inplace():
    A = np.array([[1, 2], [3, 4]])
    id_original = id(A)
    sumar_fila_multiplo(A, 0, 1, -1)
    assert id(A) == id_original


# ---------------------------------------------------------------------
# Ejercicio 11: esDiagonalmenteDominante(A) -> True si A es cuadrada y
# estrictamente diagonalmente dominante.
# ---------------------------------------------------------------------

def test_ej11_esDiagonalmenteDominante_true():
    A = np.array([[5, 1, 1],
                  [1, 4, 1],
                  [1, 1, 3]])
    assert esDiagonalmenteDominante(A) is True


def test_ej11_esDiagonalmenteDominante_false():
    A = np.array([[1, 2, 1],
                  [1, 4, 1],
                  [1, 1, 3]])
    assert esDiagonalmenteDominante(A) is False


# ---------------------------------------------------------------------
# Ejercicio 12: matrizCirculante(v) -> primer fila = v, cada fila
# siguiente es una permutacion ciclica hacia la derecha de la anterior.
# ---------------------------------------------------------------------

def test_ej12_matrizCirculante():
    v = np.array([1, 2, 3])
    esperado = np.array([[1, 2, 3],
                         [3, 1, 2],
                         [2, 3, 1]])
    np.testing.assert_array_equal(matrizCirculante(v), esperado)


# ---------------------------------------------------------------------
# Ejercicio 13: matrizVandermonde(v) -> fila i-esima = potencia
# (i-1)-esima de los elementos de v.
# ---------------------------------------------------------------------

def test_ej13_matrizVandermonde():
    v = np.array([1, 2, 3])
    esperado = np.array([[1, 1, 1],
                         [1, 2, 3],
                         [1, 4, 9]])
    np.testing.assert_array_equal(matrizVandermonde(v), esperado)


# ---------------------------------------------------------------------
# Ejercicio 14: numeroAureo(n) -> estima phi como F(k+1)/F(k) usando la
# formulacion matricial de Fibonacci.
# ---------------------------------------------------------------------

def test_ej14_numeroAureo_converge():
    phi = (1 + 5 ** 0.5) / 2  # 1.6180339887...
    estimado = numeroAureo(30)
    assert estimado == pytest.approx(phi, abs=1e-5)


# ---------------------------------------------------------------------
# Ejercicio 15: matrizFiboncacci(n) -> a_ij = F(i+j), F0=0, F1=1.
# ---------------------------------------------------------------------

def test_ej15_matrizFiboncacci():
    n = 4
    esperado = np.array([[fibonacci(i + j) for j in range(n)] for i in range(n)])
    np.testing.assert_array_equal(matrizFiboncacci(n), esperado)


# ---------------------------------------------------------------------
# Ejercicio 16: matrizHilbert(n) -> h_ij = 1 / (i+j+1).
# ---------------------------------------------------------------------

def test_ej16_matrizHilbert():
    n = 3
    esperado = np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])
    np.testing.assert_allclose(matrizHilbert(n), esperado)


# ---------------------------------------------------------------------
# Ejercicio 17: rutina que evalua, en 100 puntos equiespaciados entre
# -1 y 1, los polinomios:
#   x^5 - x^4 + x^3 - x^2 + x - 1
#   x^2 + 3
#   x^10 - 2
#
# La consigna no fija una firma/formato de retorno concreto. Se asume
# que evaluarPolinomios() devuelve una secuencia de 3 arrays (uno por
# polinomio), cada uno evaluado sobre np.linspace(-1, 1, 100). Si al
# implementarla se elige otra interfaz, ajustar este test.
# ---------------------------------------------------------------------

def test_ej17_evaluarPolinomios():
    resultado = evaluarPolinomios()
    assert resultado is not None
    p1, p2, p3 = resultado

    x = np.linspace(-1, 1, 100)
    esperado_p1 = x**5 - x**4 + x**3 - x**2 + x - 1
    esperado_p2 = x**2 + 3
    esperado_p3 = x**10 - 2

    assert len(p1) == len(x)
    np.testing.assert_allclose(p1, esperado_p1)
    np.testing.assert_allclose(p2, esperado_p2)
    np.testing.assert_allclose(p3, esperado_p3)


# ---------------------------------------------------------------------
# Ejercicio 18: row_echelon(A) modificado con pivoteo parcial: en cada
# pivot, si hay otro elemento de la misma columna con modulo mayor, se
# intercambian las filas antes de eliminar.
#
# La consigna asume una funcion row_echelon preexistente (de otra
# practica) que aca no esta definida todavia; se testea el contrato
# esperado: devuelve la forma escalonada de A, usando pivoteo parcial
# para evitar pivotes chicos/nulos.
# ---------------------------------------------------------------------

def test_ej18_row_echelon_forma_triangular():
    A = np.array([[2.0, 1.0], [4.0, 3.0]])
    resultado = row_echelon(A.copy())
    assert resultado is not None
    # Forma escalonada: por debajo de la diagonal debe quedar (aprox) 0.
    assert resultado[1][0] == pytest.approx(0.0, abs=1e-9)


def test_ej18_row_echelon_pivotea_para_evitar_pivot_cero():
    # Sin pivoteo, el primer pivote es 0 y la eliminacion falla
    # (division por cero). Con pivoteo parcial se intercambian las
    # filas y el calculo se completa con exito.
    A = np.array([[0.0, 1.0], [1.0, 1.0]])
    resultado = row_echelon(A.copy())
    assert resultado is not None
    assert np.all(np.isfinite(resultado))
    assert resultado[1][0] == pytest.approx(0.0, abs=1e-9)
