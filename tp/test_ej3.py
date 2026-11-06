"""Test sintetico del Ejercicio 3 (calibracion con puntos coplanares).

Se parte de una K, R y t conocidas, se proyectan las 77 esquinas del tablero,
y se verifica que resolverH -> armarA -> calcularMatrizK2D -> obtenerPose2D
recuperan los valores reales (sin ruido: casi exactos; con ruido: errores chicos).

Uso:  python tp/test_ej3.py   (desde la raiz del repo)  o  python test_ej3.py  (desde tp/)
Solo para verificar: usa @ y np.linalg, que no estan permitidos en el codigo entregable.
"""
import os
import sys
import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))   # modulo alc
sys.path.insert(0, AQUI)                    # calibracion.py
from calibracion import resolverH, armarA, calcularMatrizK2D, obtenerPose2D

K_real = np.array([[800., 0., 512.], [0., 810., 288.], [0., 0., 1.]])
objPts = np.array([[30.0 * i, 30.0 * j] for j in range(7) for i in range(11)])


def rotacion(a, b, c):
    """Rotacion con angulos de Euler (a, b, c) alrededor de x, y, z."""
    Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def correr(ruido, rng):
    """Simula 10 vistas con `ruido` (px) y devuelve K estimada y errores de R y t (vista 0)."""
    Hlst, poses = [], []
    for _ in range(10):
        R = rotacion(*rng.uniform(-0.5, 0.5, 3))
        t = np.array([-150., -100., 600. + rng.uniform(0, 200)])
        H = K_real @ np.column_stack([R[:, 0], R[:, 1], t])
        P = np.hstack([objPts, np.ones((77, 1))]) @ H.T
        uv = P[:, :2] / P[:, 2:] + rng.normal(0, ruido, (77, 2))
        Hlst.append(resolverH(uv, objPts))
        poses.append((R, t))
    K = calcularMatrizK2D(armarA(Hlst))
    R, t = obtenerPose2D(Hlst[0], K)
    return K, np.abs(R - poses[0][0]).max(), np.abs(t - poses[0][1]).max()


def proyectarH(H, pts):
    """Proyecta pts (n x 2) del tablero con la homografia H."""
    P = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return P[:, :2] / P[:, 2:]


if __name__ == "__main__":
    import matplotlib.image as mpimg
    from calibracion import obtenerQ2d
    from esquinas import esquinasTablero

    np.set_printoptions(precision=3, suppress=True)
    fallos = []

    def chequear(nombre, ok, detalle=""):
        print(("OK    " if ok else "FALLA ") + nombre + (" - " + detalle if detalle else ""))
        if not ok:
            fallos.append(nombre)

    rng = np.random.default_rng(0)

    # --- 1) obtenerQ2d
    R0 = rotacion(0.2, -0.3, 0.1)
    t0 = np.array([-150., -100., 650.])
    H0 = K_real @ np.column_stack([R0[:, 0], R0[:, 1], t0])
    uv0 = proyectarH(H0, objPts)
    Q = obtenerQ2d(uv0, objPts)
    chequear("Q es 154 x 9", Q.shape == (154, 9))
    h0 = (H0 / np.linalg.norm(H0)).reshape(-1)
    chequear("Q h_verdadera = 0", np.abs(Q @ h0).max() < 1e-9, "%.1e" % np.abs(Q @ h0).max())
    chequear("rango(Q) = 8", np.linalg.matrix_rank(Q, tol=1e-9 * np.linalg.norm(Q, 2)) == 8)
    obj3 = np.hstack([objPts, np.zeros((77, 1))])
    chequear("objPts n x 3 con W=0 da la misma Q", np.abs(obtenerQ2d(uv0, obj3) - Q).max() == 0)
    obj3[5, 2] = 1.0
    try:
        obtenerQ2d(uv0, obj3)
        chequear("W != 0 lanza ValueError", False)
    except ValueError:
        chequear("W != 0 lanza ValueError", True)

    # --- 2) resolverH exacta, con y sin Hartley
    for norm in [True, False]:
        H = resolverH(uv0, objPts, normalizar=norm)
        err = np.abs(proyectarH(H, objPts) - uv0).max()
        chequear("resolverH normalizar=%s: reproyeccion exacta" % norm, err < 1e-6, "%.1e px" % err)

    # --- 3) pipeline completo con datos sinteticos
    for ruido, tolK in [(0.0, 1e-4), (0.3, 5.0)]:
        K, errR, errt = correr(ruido, rng)
        chequear("sintetico ruido %.1f: K" % ruido, np.abs(K - K_real).max() < tolK,
                 "max|K-K_real| = %.2e" % np.abs(K - K_real).max())
        chequear("sintetico ruido %.1f: R y t" % ruido, errR < (1e-6 if ruido == 0 else 0.05)
                 and errt < (1e-3 if ruido == 0 else 10), "R %.1e, t %.1e" % (errR, errt))

    # --- 4) datos reales: 20 imagenes
    carpetaBase = os.path.join(AQUI, "template-alumnos-20261027")
    nombre = [d for d in os.listdir(carpetaBase) if d.strip() == "calib"][0]
    carpeta = os.path.join(carpetaBase, nombre, "coplanar")
    Hlst, puntos = [], []
    deteccionesOk = True
    errH = 0.0
    for k in range(1, 21):
        P = esquinasTablero(mpimg.imread(os.path.join(carpeta, "Im_R_%d.png" % k)))
        if P is None or P.shape != (77, 2):
            deteccionesOk = False
            break
        H = resolverH(P, objPts)
        errH = max(errH, np.abs(proyectarH(H, objPts) - P).max())
        puntos.append(P)
        Hlst.append(H)
    chequear("deteccion: 77 esquinas en las 20 imagenes", deteccionesOk)
    chequear("deteccion/orden: H de cada vista reproyecta las esquinas < 1 px", errH < 1.0,
             "max %.2f px" % errH)
    if deteccionesOk:
        K = calcularMatrizK2D(armarA(Hlst))
        print(K)
        chequear("K real: triangular, K33 = 1", K[1, 0] == 0 and K[2, 0] == 0 and K[2, 1] == 0
                 and K[2, 2] == 1)
        chequear("K real: alfas positivos y parecidos (dif < 5%)", K[0, 0] > 0 and K[1, 1] > 0
                 and abs(K[0, 0] - K[1, 1]) / K[0, 0] < 0.05)
        chequear("K real: gamma chico, (u0, v0) cerca del centro (512, 288)",
                 abs(K[0, 1]) < 0.01 * K[0, 0] and abs(K[0, 2] - 512) < 60 and abs(K[1, 2] - 288) < 60)
        okPoses = True
        peor = 0.0
        for j in range(20):
            R, t = obtenerPose2D(Hlst[j], K)
            err = np.abs(proyectarH(K @ np.column_stack([R[:, 0], R[:, 1], t]), objPts) - puntos[j]).max()
            peor = max(peor, err)
            okPoses = okPoses and np.abs(R @ R.T - np.eye(3)).max() < 1e-6 \
                and abs(np.linalg.det(R) - 1) < 1e-6 and t[2] > 0
        chequear("20 poses: R ortonormal, det=1, t3>0", okPoses)
        chequear("20 poses: reproyeccion maxima < 1.5 px", peor < 1.5, "max %.2f px" % peor)

    print("\nRESULTADO:", "TODO OK" if not fallos else "FALLARON: " + ", ".join(fallos))
    sys.exit(1 if fallos else 0)
