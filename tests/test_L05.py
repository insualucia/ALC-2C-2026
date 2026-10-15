"""Tests de L05 (QR). Original: material/L05/tests_L05.py"""
import numpy as np
import pytest

from alc import QR_con_GS, QR_con_HH, calculaQR

A2 = np.array([[1., 2.],
               [3., 4.]])

A3 = np.array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 0.]])

A4 = np.array([[2., 0., 1., 3.],
               [0., 1., 4., 1.],
               [1., 0., 2., 0.],
               [3., 1., 0., 2.]])


def check_QR(Q, R, A, tol=1e-10):
    # Comprueba ortogonalidad y reconstruccion
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
    assert np.allclose(Q @ R, A, atol=tol)


@pytest.mark.parametrize("A", [A2, A3, A4])
def test_QR_con_GS(A):
    Q, R = QR_con_GS(A)
    check_QR(Q, R, A)


# El original testea A2 con QR_con_GS en esta seccion (probable error de tipeo);
# aca se usa QR_con_HH para las tres.
@pytest.mark.parametrize("A", [A2, A3, A4])
def test_QR_con_HH(A):
    Q, R = QR_con_HH(A)
    check_QR(Q, R, A)


@pytest.mark.parametrize("A, metodo", [(A2, 'RH'), (A3, 'GS'), (A4, 'RH')])
def test_calculaQR(A, metodo):
    Q, R = calculaQR(A, metodo=metodo)
    check_QR(Q, R, A)
