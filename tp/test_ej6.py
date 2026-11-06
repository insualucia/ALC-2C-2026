"""Test del Ejercicio 6 (POSIT coplanar).

1) calculoLambdaMu: I = I0 + lam n, J = J0 + mu n cumplen I.J = 0 y ||I|| = ||J||.
2) Sintetico sin ruido: pose conocida -> aplicarPOSIT recupera R y t.
3) Sintetico con ruido de 0.5 px.
4) Datos reales (heli/): RMS chico, R ortonormal con det 1, y pose objetivo.

Uso:  python tp/test_ej6.py   (desde la raiz del repo)
Solo para verificar: usa @, que no esta permitido en el codigo entregable.
"""
import os
import sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
sys.path.insert(0, AQUI)
from calibracion import calculoLambdaMu, aplicarPOSIT, evaluarPose, reproyectarPose

K = np.array([[1892.9, 0, 819.5], [0, 1909.5, 615.5], [0, 0, 1.0]])
obj = np.array([[0, 0, 0], [-.175, .3, 0], [.175, .3, 0], [.175, -.3, 0], [-.175, -.3, 0]])


def rot(a, b, c):
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# 1) lambda, mu
rng = np.random.default_rng(0)
for _ in range(100):
    I0, J0, n = rng.normal(size=3), rng.normal(size=3), rng.normal(size=3)
    n /= np.linalg.norm(n)
    I0, J0 = I0 - (I0 @ n) * n, J0 - (J0 @ n) * n   # I0, J0 en el plano (fila de A)
    lam, mu = calculoLambdaMu(I0, J0)
    for s in (1, -1):
        I, J = I0 + s * lam * n, J0 + s * mu * n
        assert abs(I @ J) < 1e-9 and abs(I @ I - J @ J) < 1e-9
print("1) calculoLambdaMu OK")

# 2) y 3) sinteticos. POSIT coplanar es voraz (elige la rama de menor RMS en cada
# iteracion), asi que con poca perspectiva puede quedarse en la pose simetrica:
# se cuenta cuantas veces recupera la pose real.
for ruido in (0.0, 0.5):
    bien, peor_R, peor_t = 0, 0, 0
    N = 200
    for _ in range(N):
        R0 = rot(np.pi + rng.uniform(-.3, .3), rng.uniform(-.3, .3), rng.uniform(-3, 3))
        t0 = np.array([rng.uniform(-.3, .3), rng.uniform(-.3, .3), rng.uniform(2, 5)])
        pix = reproyectarPose(K, R0, t0, obj) + rng.normal(0, ruido, (5, 2))
        R, t = aplicarPOSIT(obj, pix, K, 100, 1e-10)
        eR = np.abs(R - R0).max()
        if eR < (1e-3 if ruido == 0 else 0.05):
            bien += 1
            peor_R = max(peor_R, eR)
            peor_t = max(peor_t, np.linalg.norm(t - t0) / np.linalg.norm(t0))
    print(f"2/3) ruido {ruido}: recupera la pose real en {bien}/{N}; err max R = {peor_R:.2e}, err rel max t = {peor_t:.2e}")
    assert bien >= 0.85 * N

# 4) reales
pix = np.array([[523.9, 1006.0], [402, 1097], [393, 945], [644, 913], [657, 1062]], float)
R, t = aplicarPOSIT(obj, pix, K, 100, 1e-10)
print(evaluarPose(R, t, obj, pix, K), "\nR=\n", R, "\nt=", t)
assert abs(np.linalg.det(R) - 1) < 1e-6 and np.abs(R @ R.T - np.eye(3)).max() < 1e-6
print("OK")
