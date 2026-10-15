"""Tests de L01 (Numeros de maquina). Original: material/L01/tests01.py"""
import numpy as np

from alc import error, error_relativo, matricesIguales


def sonIguales(x, y, atol=1e-08):
    """Auxiliar provista en el enunciado para testear error."""
    return np.allclose(error(x, y), 0, atol=atol)


def test_error():
    assert not sonIguales(1, 1.1)
    assert sonIguales(1, 1 + np.finfo('float64').eps)
    assert not sonIguales(1, 1 + np.finfo('float32').eps)
    assert not sonIguales(np.float16(1), np.float16(1) + np.finfo('float32').eps)
    assert sonIguales(np.float16(1), np.float16(1) + np.finfo('float16').eps, atol=1e-3)


def test_error_relativo():
    assert np.allclose(error_relativo(1, 1.1), 0.1)
    assert np.allclose(error_relativo(2, 1), 0.5)
    assert np.allclose(error_relativo(-1, -1), 0)
    assert np.allclose(error_relativo(1, -1), 2)


def test_matricesIguales():
    assert matricesIguales(np.diag([1, 1]), np.eye(2))
    assert matricesIguales(np.linalg.inv(np.array([[1, 2], [3, 4]])) @ np.array([[1, 2], [3, 4]]), np.eye(2))
    assert not matricesIguales(np.array([[1, 2], [3, 4]]).T, np.array([[1, 2], [3, 4]]))
