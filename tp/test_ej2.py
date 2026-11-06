"""Tests del Ejercicio 2 (calibracion con puntos 3D, DLT).

Uso:  python tp/test_ej2.py  (desde la raiz)  o  python test_ej2.py  (desde tp/)
Usa @ y np.linalg solo para verificar; no es parte del entregable.
"""
import os
import sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
sys.path.insert(0, AQUI)
from calibracion import (obtenerQ3d, resolverDLT3D,
                         devolverMatricesCalibracion3d, verificarCalibracion3d)

carpeta = os.path.join(AQUI, "template-alumnos-20261027",
                       [d for d in os.listdir(os.path.join(AQUI, "template-alumnos-20261027"))
                        if d.strip() == "calib"][0], "3d")
objPts = np.loadtxt(os.path.join(carpeta, "p_W_corners.txt"), delimiter=",")
detecciones = np.loadtxt(os.path.join(carpeta, "detected_corners.txt"))
fallos = []


def chequear(nombre, ok, detalle=""):
    print(("OK    " if ok else "FALLA ") + nombre + (" - " + detalle if detalle else ""))
    if not ok:
        fallos.append(nombre)


def proyectar(K, R, t, pts):
    P = np.hstack([pts, np.ones((len(pts), 1))]) @ (K @ np.hstack([R, t.reshape(3, 1)])).T
    return P[:, :2] / P[:, 2:]


# --- 1) Q: forma y consistencia con m verdadera
Kr = np.array([[800., 0., 512.], [0., 810., 288.], [0., 0., 1.]])
th = 0.3
R0 = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]) @ \
     np.array([[1, 0, 0], [0, np.cos(.2), -np.sin(.2)], [0, np.sin(.2), np.cos(.2)]])
t0 = np.array([5., -3., 80.])
uv = proyectar(Kr, R0, t0, objPts)
Q = obtenerQ3d(uv, objPts)
M0 = Kr @ np.hstack([R0, t0.reshape(3, 1)])
chequear("Q es 24 x 12", Q.shape == (24, 12))
chequear("Q m_verdadera = 0", np.abs(Q @ M0.reshape(-1)).max() / np.abs(M0).max() < 1e-9,
         "%.1e" % (np.abs(Q @ M0.reshape(-1)).max() / np.abs(M0).max()))
chequear("rango(Q) = 11", np.linalg.matrix_rank(Q, tol=1e-8 * np.linalg.norm(Q, 2)) == 11)

# --- 2) datos exactos: recupera K, R, t
M = resolverDLT3D(Q)
R, t, K = devolverMatricesCalibracion3d(M)
chequear("datos exactos: K", np.abs(K - Kr).max() < 1e-4, "%.1e" % np.abs(K - Kr).max())
chequear("datos exactos: R", np.abs(R - R0).max() < 1e-6, "%.1e" % np.abs(R - R0).max())
chequear("datos exactos: t", np.abs(t - t0).max() < 1e-4, "%.1e" % np.abs(t - t0).max())
R2, t2, K2 = devolverMatricesCalibracion3d(-M)
chequear("invariante al signo de M", np.abs(K2 - K).max() < 1e-9 and np.abs(R2 - R).max() < 1e-9
         and np.abs(t2 - t).max() < 1e-9)

# --- 3) restricciones opcionales
_, _, Kg = devolverMatricesCalibracion3d(M, imponerGammaCero=True)
_, _, Ka = devolverMatricesCalibracion3d(M, imponerAlfasIguales=True)
chequear("gamma = 0 impuesto", Kg[0, 1] == 0)
chequear("alfa_u = alfa_v impuesto", Ka[0, 0] == Ka[1, 1])

# --- 4) datos reales: las 210 detecciones
maxErr = 0.0
peor = None
okTodo = True
for f in range(len(detecciones)):
    img = detecciones[f].reshape(12, 2)
    M = resolverDLT3D(obtenerQ3d(img, objPts))
    R, t, K = devolverMatricesCalibracion3d(M)
    c = verificarCalibracion3d(M, R, t, K, img, objPts)
    okTodo = okTodo and c["ortogonalidad"] < 1e-12 and abs(c["det_R"] - 1) < 1e-9 \
        and c["K_triangular"] == 0 and c["dif_M"] < 1e-9 and c["delante_camara"]
    if c["error_reproyeccion"] > maxErr:
        maxErr, peor = c["error_reproyeccion"], f
chequear("210 detecciones: R ortonormal, det=1, K triangular, K[R|t]=M, puntos delante", okTodo)
chequear("210 detecciones: error de reproyeccion maximo < 3 px", maxErr < 3,
         "max %.2f px (deteccion %d)" % (maxErr, peor))

print("\nRESULTADO:", "TODO OK" if not fallos else "FALLARON: " + ", ".join(fallos))
sys.exit(1 if fallos else 0)
