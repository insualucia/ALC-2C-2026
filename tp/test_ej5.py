"""Test del Ejercicio 5.a (intrinsecos con gamma = 0, pixeles cuadrados y centro fijo).

1) Sintetico sin ruido: K = (f, f, centro), puntos coplanares (Z = 0) vistos con
   poses distintas -> calcularKRestringida recupera f casi exacto, residuo ~ 0.
2) Sintetico con ruido de 0.5 px: f con error chico.
3) Datos reales (world_img_pts.csv, Z ignorado): f del orden de 3500 px.

Uso:  python tp/test_ej5.py   (desde la raiz del repo)
Solo para verificar: usa @, que no esta permitido en el codigo entregable.
"""
import os
import sys
import glob
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
sys.path.insert(0, AQUI)
from calibracion import (resolverH, calcularKRestringida, obtenerPose2D, reproyectarPose,
                         velocidadMedia, velocidadesLocales)

CENTRO = (800.0, 600.0)
F_REAL = 3400.0
K_real = np.array([[F_REAL, 0, CENTRO[0]], [0, F_REAL, CENTRO[1]], [0, 0, 1.0]])
objPts = np.array([[-4., 3.], [4., 0.], [6.5, 3.], [-5., 14.], [-5., 76.], [0., 30.], [3., 50.]])


def rotacion(a, b, c):
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def vistas(ruido, rng, n=6):
    Hlst = []
    for _ in range(n):
        R = rotacion(*rng.uniform(-0.4, 0.4, 3))
        t = np.array([0., 0., 60. + rng.uniform(0, 40)])
        H = K_real @ np.column_stack([R[:, 0], R[:, 1], t])
        P = np.hstack([objPts, np.ones((len(objPts), 1))]) @ H.T
        uv = P[:, :2] / P[:, 2:] + rng.normal(0, ruido, (len(objPts), 2))
        Hlst.append(resolverH(uv, objPts))
    return Hlst


def main():
    rng = np.random.default_rng(0)
    # 1) sin ruido
    K, res = calcularKRestringida(vistas(0.0, rng), CENTRO)
    print("sin ruido: f =", K[0, 0], " residuo =", res)
    assert abs(K[0, 0] - F_REAL) < 1e-3 * F_REAL
    assert res < 1e-6
    assert K[0, 1] == 0 and K[1, 0] == 0 and K[0, 0] == K[1, 1]
    assert (K[0, 2], K[1, 2]) == CENTRO
    # 2) con ruido
    fs = []
    for _ in range(20):
        K, _ = calcularKRestringida(vistas(0.5, rng), CENTRO)
        fs.append(K[0, 0])
    fs = np.array(fs)
    print("ruido 0.5 px: f medio = %.0f, desvio = %.0f" % (fs.mean(), fs.std()))
    assert abs(fs.mean() - F_REAL) < 0.1 * F_REAL
    # 3) datos reales
    ruta = glob.glob(os.path.join(AQUI, "**", "world_img_pts.csv"), recursive=True)[0]
    d = np.loadtxt(ruta, delimiter=";", skiprows=1)
    Hs = [resolverH(d[:, 4 + 2 * k:6 + 2 * k], d[:, 1:3]) for k in range(4)]
    K, res = calcularKRestringida(Hs, CENTRO)
    print("datos reales: f = %.0f, residuo relativo = %.2f" % (K[0, 0], res))
    assert 3000 < K[0, 0] < 4000
    # 4) Ej. 5.b sintetico: pose conocida, 51 puntos del piso (Z = 0), sin ruido
    R_ver = np.array([[0.98859, 0.15061, 0.00225], [0.03295, -0.20165, -0.97890], [-0.14698, 0.96781, -0.20431]])
    U, _, Vt = np.linalg.svd(R_ver)
    R_ver = U @ Vt                       # rotacion exacta
    t_ver = np.array([-3.3, -1.0, 23.0])
    K2 = np.array([[2000., 0, 800.], [0, 2000., 600.], [0, 0, 1.]])
    mundo = np.array([[x, y, 0.] for x in (-6., -3., 0., 3., 6., 9.) for y in (-12., 0., 3., 14., 21., 34., 50., 76., 60.)])
    P = (K2 @ (R_ver @ mundo.T + t_ver[:, None])).T
    pix = P[:, :2] / P[:, 2:]
    H = resolverH(pix, mundo[:, :2])
    K, res = calcularKRestringida([H], CENTRO)
    R, t = obtenerPose2D(H, K)
    rp = reproyectarPose(K, R, t, mundo)
    print("5.b sintetico: f = %.4f, |R-R_ver| = %.1e, |t-t_ver| = %.1e, error reproy. max = %.1e px"
          % (K[0, 0], np.abs(R - R_ver).max(), np.abs(t - t_ver).max(), np.abs(rp - pix).max()))
    assert abs(K[0, 0] - 2000.) < 1e-4
    assert np.abs(R - R_ver).max() < 1e-6 and np.abs(t - t_ver).max() < 1e-4
    assert np.abs(rp - pix).max() < 1e-4
    # 5) datos reales del 5.b (51 puntos, Z = 0)
    D = np.loadtxt(glob.glob(os.path.join(AQUI, "**", "gs_10_09_point_correspondences_v5.txt"), recursive=True)[0])
    mundo, pix = D[1:4].T, D[4:6].T
    H = resolverH(pix, mundo[:, :2])
    K, res = calcularKRestringida([H], CENTRO)
    R, t = obtenerPose2D(H, K)
    e = np.sqrt(np.sum((reproyectarPose(K, R, t, mundo) - pix) ** 2, axis=1))
    print("5.b real: f = %.0f, residuo = %.3f, error reproy. medio %.1f / max %.1f px" % (K[0, 0], res, e.mean(), e.max()))
    assert 1800 < K[0, 0] < 2200 and e.max() < 40
    # 6) Ej. 5.c sintetico: movimiento uniforme (3, -14) m/s con ruido de 0.3 m en la posicion
    t = np.cumsum(rng.uniform(0.03, 0.25, 40))
    P = np.column_stack([1.0 + 3.0 * t, 40.0 - 14.0 * t]) + rng.normal(0, 0.3, (40, 2))
    v, (vx, vy) = velocidadMedia(t, P)
    vl = velocidadesLocales(t, P, k=5)
    print("5.c sintetico: v = %.2f (real 14.32), (vx, vy) = (%.2f, %.2f); locales: media %.2f" % (v, vx, vy, vl.mean()))
    assert abs(v - np.hypot(3., 14.)) < 0.3 and abs(vx - 3.) < 0.3 and abs(vy + 14.) < 0.3
    assert abs(vl.mean() - np.hypot(3., 14.)) < 1.0
    # sin ruido: exacto
    v0, _ = velocidadMedia(t, np.column_stack([1.0 + 3.0 * t, 40.0 - 14.0 * t]))
    assert abs(v0 - np.hypot(3., 14.)) < 1e-9
    print("TODO OK")


if __name__ == "__main__":
    main()
