"""Tests del Ejercicio 4 (homografia de la cancha: distancia, circulos, offside).

Uso:  python tp/test_ej4.py  (desde la raiz)  o  python test_ej4.py  (desde tp/)
Usa np.linalg y @ solo para verificar; no es parte del entregable.
"""
import os
import sys
import numpy as np
import matplotlib.image as mpimg

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(AQUI))
sys.path.insert(0, AQUI)
from calibracion import resolverH, proyectarPuntos, circuloEnPlano
from alc import inversa

fallos = []


def chequear(nombre, ok, detalle=""):
    print(("OK    " if ok else "FALLA ") + nombre + (" - " + detalle if detalle else ""))
    if not ok:
        fallos.append(nombre)


# --- datos de las fotos (los mismos que usa el notebook)
mitadCuerda = np.sqrt(9.15 ** 2 - 5.5 ** 2)
pxGol = np.array([[196, 171], [841, 190], [156, 231], [868, 251], [147, 406], [852, 422]], float)
canchaGol = np.array([[0, 34 - 9.16], [0, 34 + 9.16], [5.5, 34 - 9.16], [5.5, 34 + 9.16],
                      [16.5, 34 - mitadCuerda], [16.5, 34 + mitadCuerda]])
pxOff = np.array([[736, 316], [735, 757], [1230, 391], [1368, 576], [736, 557], [1305, 236]], float)
canchaOff = np.array([[16.5, 13.84], [16.5, 54.16], [5.5, 24.84], [5.5, 43.16],
                      [16.5, 34 + mitadCuerda], [0, 0]])

# --- 1) proyectarPuntos
H0 = np.array([[2., 0.1, 1.], [0.2, 3., 2.], [0.0005, 0.001, 1.]])
p = np.array([[1., 1.], [2., 3.], [10., -4.]])
manual = np.array([(H0 @ np.r_[q, 1])[:2] / (H0 @ np.r_[q, 1])[2] for q in p])
chequear("proyectarPuntos coincide con el calculo a mano", np.abs(proyectarPuntos(H0, p) - manual).max() < 1e-12)
ida_vuelta = proyectarPuntos(inversa(H0), proyectarPuntos(H0, p))
chequear("H^-1 deshace a H", np.abs(ida_vuelta - p).max() < 1e-9, "%.1e" % np.abs(ida_vuelta - p).max())
chequear("acepta un solo punto", proyectarPuntos(H0, np.array([1., 1.])).shape == (1, 2))

# --- 2) circuloEnPlano
c = circuloEnPlano((3., 4.), 2., 50)
radios = np.sqrt((c[:, 0] - 3) ** 2 + (c[:, 1] - 4) ** 2)
chequear("circulo: radio exacto", np.abs(radios - 2).max() < 1e-12)
chequear("circulo: cerrado y de 2n+1 puntos", c.shape == (101, 2) and np.abs(c[0] - c[-1]).max() < 1e-12)

# --- 3) sintetico: H conocida, se recupera H y se mide
rng = np.random.default_rng(0)
K = np.array([[900., 0., 540.], [0., 900., 337.], [0., 0., 1.]])
th = 0.9
R = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
t = np.array([-30., 20., 70.])
Hv = K @ np.column_stack([R[:, 0], R[:, 1], t])
pxS = proyectarPuntos(Hv, canchaGol)
for ruido, tol in [(0.0, 1e-6), (1.0, 0.8)]:
    H = resolverH(pxS + rng.normal(0, ruido, pxS.shape), canchaGol)
    punto = np.array([[22.0, 39.0]])
    pie = proyectarPuntos(Hv, punto)
    U, V = proyectarPuntos(inversa(H), pie)[0]
    chequear("sintetico ruido %.0f px: U medido" % ruido, abs(U - 22.0) < tol, "U = %.4f (real 22)" % U)
H = resolverH(pxS, canchaGol)
cir = circuloEnPlano((22., 39.), 1.0, 60)
vuelta = proyectarPuntos(inversa(H), proyectarPuntos(H, cir))
chequear("sintetico: el circulo proyectado vuelve al de 1 m",
         np.abs(np.sqrt(np.sum((vuelta - [22., 39.]) ** 2, axis=1)) - 1).max() < 1e-6)

# --- 4) fotos reales
carpeta = os.path.join(AQUI, "template-alumnos-20261027", "futbol")
fotoGol = mpimg.imread(os.path.join(carpeta, "test-gol-enzo.png"))[:, :, 0:3]
fotoOff = mpimg.imread(os.path.join(carpeta, "test-fuera-de-juego.png"))[:, :, 0:3]


def hayLineaBlanca(foto, px, radio=5):
    """True si cerca de px hay un pixel claro y notablemente mas claro que su entorno."""
    g = np.mean(foto, axis=2)
    u, v = int(round(px[0])), int(round(px[1]))
    ventana = g[v - radio:v + radio + 1, u - radio:u + radio + 1]
    entorno = g[v - 20:v + 21, u - 20:u + 21]
    return ventana.max() > 0.7 and ventana.max() - np.median(entorno) > 0.25


for nombre, px, cancha, foto in [("gol-enzo", pxGol, canchaGol, fotoGol), ("fuera-de-juego", pxOff, canchaOff, fotoOff)]:
    H = resolverH(px, cancha)
    err = np.sqrt(np.sum((proyectarPuntos(H, cancha) - px) ** 2, axis=1))
    chequear("%s: reproyeccion de los 6 puntos < 2.5 px" % nombre, err.max() < 2.5, "max %.2f px" % err.max())
    # punto NO usado en el ajuste: el arco mas alejado del arco del area (U = 20.15, V = 34)
    q = proyectarPuntos(H, np.array([[20.15, 34.0]]))[0]
    chequear("%s: el punto del arco (20.15, 34) cae sobre la linea blanca" % nombre,
             hayLineaBlanca(foto, q), "pixel (%.0f, %.0f)" % (q[0], q[1]))

def desvioLinea(foto, q, orientacion, r=12):
    """
    A cuantos pixeles de q esta la linea real (maximo de brillo del perfil perpendicular).
    orientacion 'h': linea casi horizontal (perfil vertical); 'v': casi vertical (perfil horizontal).
    Retorna (desvio en px, pico - mediana del perfil).
    """
    g = np.mean(foto, axis=2)
    u, v = int(round(q[0])), int(round(q[1]))
    if orientacion == "h":
        perfil = g[v - r:v + r + 1, u - 4:u + 5].mean(axis=1)
    else:
        perfil = g[v - 4:v + 5, u - r:u + r + 1].mean(axis=0)
    return int(np.argmax(perfil)) - r, perfil.max() - np.median(perfil)


# puntos de lineas (casi horizontales o verticales) que NO se usaron para ajustar H:
# la linea real tiene que estar a <= 3 px del punto proyectado. El mayor desvio (~3 px) esta en el
# medio del arco del area, en las dos fotos: una homografia no modela la distorsion de la lente.
# (El punto (16.5, 20) de la foto del offside no sirve: esta tapado por jugadores.)
extra = [("gol-enzo", pxGol, canchaGol, fotoGol, [
            ("frente area chica (5.5, 30)", (5.5, 30), "h"), ("frente area chica (5.5, 38)", (5.5, 38), "h"),
            ("arco (20.15, 34)", (20.15, 34), "h")]),
         ("fuera-de-juego", pxOff, canchaOff, fotoOff, [
            ("linea area grande (16.5, 48)", (16.5, 48), "v"), ("linea area grande (16.5, 52)", (16.5, 52), "v"),
            ("lado area grande (10, 54.16)", (10, 54.16), "h"), ("lado area grande (8, 13.84)", (8, 13.84), "h"),
            ("lado area chica (3, 43.16)", (3, 43.16), "h"), ("lado area chica (3, 24.84)", (3, 24.84), "h"),
            ("linea de banda (30, 0)", (30, 0), "h"), ("linea de banda (20, 0)", (20, 0), "h")])]
for nombre, px, cancha, foto, puntos in extra:
    H = resolverH(px, cancha)
    for etiqueta, (U, V), orient in puntos:
        q = proyectarPuntos(H, np.array([[U, V]]))[0]
        d, contraste = desvioLinea(foto, q, orient)
        chequear("%s: %s sobre su linea" % (nombre, etiqueta), abs(d) <= 3 and contraste > 0.05,
                 "desvio %+d px, contraste %.2f" % (d, contraste))

H = resolverH(pxGol, canchaGol)
U, V = proyectarPuntos(inversa(H), np.array([[807., 546.]]))[0]
chequear("gol-enzo: Enzo a 21-24 m de la linea de gol", 21 < U < 24, "U = %.1f m" % U)

H2inv = inversa(resolverH(pxOff, canchaOff))
U0 = proyectarPuntos(H2inv, np.array([[822., 332.]]))[0][0]
chequear("fuera-de-juego: linea de offside en U ~ 14.4 m", abs(U0 - 14.4) < 0.3, "U0 = %.2f m" % U0)

print("\nRESULTADO:", "TODO OK" if not fallos else "FALLARON: " + ", ".join(fallos))
sys.exit(1 if fallos else 0)
