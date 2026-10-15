"""
Tests de L03 (Normas y condicion). Original: material/L03/Tests L03-Normas.py

Diferencias con el original, porque contradice la consigna del PDF:
  - El original llama condMC(A, 2, 10000) (con Np) y la consigna dice
    condMC(A, p). condMC acepta Np como parametro opcional, asi andan las dos.
  - El original llama condExacta; la consigna dice condExacto. Se usa
    el nombre de la consigna.
"""
import numpy as np

from alc import norma, normaliza, normaMatMC, normaExacta, condMC, condExacto


def test_norma():
    assert np.allclose(norma(np.array([1, 1]), 2), np.sqrt(2))
    assert np.allclose(norma(np.array([1] * 10), 2), np.sqrt(10))
    assert norma(np.random.rand(10), 2) <= np.sqrt(10)
    assert norma(np.random.rand(10), 2) >= 0


def test_normaliza():
    for x in normaliza([np.array([1] * k) for k in range(1, 11)], 2):
        assert np.allclose(norma(x, 2), 1)
    # En el original este bloque solo hace print (no es un assert):
    # normalizar en norma 1 no deja norma 2 igual a 1.
    for x in normaliza([np.array([1] * k) for k in range(2, 11)], 1):
        assert not np.allclose(norma(x, 2), 1)
    for x in normaliza([np.random.rand(k) for k in range(1, 11)], 'inf'):
        assert np.allclose(norma(x, 'inf'), 1)


def test_normaExacta():
    assert np.allclose(normaExacta(np.array([[1, -1], [-1, -1]]), 1), 2)
    assert np.allclose(normaExacta(np.array([[1, -2], [-3, -4]]), 1), 6)
    assert np.allclose(normaExacta(np.array([[1, -2], [-3, -4]]), 'inf'), 7)
    assert normaExacta(np.array([[1, -2], [-3, -4]]), 2) is None
    assert normaExacta(np.random.random((10, 10)), 1) <= 10
    assert normaExacta(np.random.random((4, 4)), 'inf') <= 4


def test_normaMatMC():
    nMC = normaMatMC(A=np.eye(2), q=2, p=1, Np=100000)
    assert np.allclose(nMC[0], 1, atol=1e-3)
    assert np.allclose(np.abs(nMC[1][0]), 1, atol=1e-3) or np.allclose(np.abs(nMC[1][1]), 1, atol=1e-3)
    assert np.allclose(np.abs(nMC[1][0]), 0, atol=1e-3) or np.allclose(np.abs(nMC[1][1]), 0, atol=1e-3)

    nMC = normaMatMC(A=np.eye(2), q=2, p='inf', Np=100000)
    assert np.allclose(nMC[0], np.sqrt(2), atol=1e-3)
    assert np.allclose(np.abs(nMC[1][0]), 1, atol=1e-3) and np.allclose(np.abs(nMC[1][1]), 1, atol=1e-3)

    A = np.array([[1, 2], [3, 4]])
    nMC = normaMatMC(A=A, q='inf', p='inf', Np=1000000)
    assert np.allclose(nMC[0], normaExacta(A, 'inf'), rtol=2e-1)


def test_condMC():
    A = np.array([[1, 1], [0, 1]])
    A_ = np.linalg.solve(A, np.eye(A.shape[0]))
    normaA = normaMatMC(A, 2, 2, 10000)
    normaA_ = normaMatMC(A_, 2, 2, 10000)
    condA = condMC(A, 2, 10000)
    assert np.allclose(normaA[0] * normaA_[0], condA, atol=1e-3)

    A = np.array([[3, 2], [4, 1]])
    A_ = np.linalg.solve(A, np.eye(A.shape[0]))
    normaA = normaMatMC(A, 2, 2, 10000)
    normaA_ = normaMatMC(A_, 2, 2, 10000)
    condA = condMC(A, 2, 10000)
    assert np.allclose(normaA[0] * normaA_[0], condA, atol=1e-3)


def test_condExacto():
    A = np.random.rand(10, 10)
    A_ = np.linalg.solve(A, np.eye(A.shape[0]))
    normaA = normaExacta(A, 1)
    normaA_ = normaExacta(A_, 1)
    condA = condExacto(A, 1)
    assert np.allclose(normaA * normaA_, condA)

    A = np.random.rand(10, 10)
    A_ = np.linalg.solve(A, np.eye(A.shape[0]))
    normaA = normaExacta(A, 'inf')
    normaA_ = normaExacta(A_, 'inf')
    condA = condExacto(A, 'inf')
    assert np.allclose(normaA * normaA_, condA)
