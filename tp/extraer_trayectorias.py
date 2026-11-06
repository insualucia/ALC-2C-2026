"""Extrae las trayectorias en pixeles de los tres agentes del Ej. 5.c.

Es una herramienta de MEDICION (equivalente a marcar los pixeles a mano, como en el
Ej. 4): usa scipy.ndimage y np.median, que no estan permitidos en el codigo entregable.
El resultado queda en  template-alumnos-20261027/salguero/trayectorias_pix.csv  y el
notebook solo lee ese archivo. No es parte de la entrega.

Uso (desde la raiz del repo):  python tp/extraer_trayectorias.py

Que mide en cada frame (la camara es fija):
  - mujer de buzo rojo: punto de apoyo = parte mas baja de los zapatos rojos
    (mascara de color; x = promedio de los pixeles rojos de las ultimas 10 filas).
  - taxi: centro de la base del recuadro de la mancha en movimiento (resta con la
    mediana de todos los frames), seguido de frame en frame; frames 16 a 43 sin el 26.
  - camioneta: centro de la base del recuadro de la mancha oscura (max(R,G,B) < 75),
    con filtro de forma, seguida de frame en frame; frames 25 a 45.
  Para los vehiculos, la base del recuadro incluye parte de la sombra: es una
  aproximacion del punto de apoyo en el piso.
"""
import os
import glob
import numpy as np
import matplotlib.image as mpimg
from scipy import ndimage as ndi

AQUI = os.path.dirname(os.path.abspath(__file__))
PLANTILLA = os.path.join(AQUI, "template-alumnos-20261027")
FRAMES = glob.glob(os.path.join(PLANTILLA, "**", "gs_dataset", "img_*.jpg"), recursive=True)
SALIDA = os.path.join(PLANTILLA, "salguero", "trayectorias_pix.csv")


def manchas(mascara, area_min):
    """Recuadros (area, x0, x1, y0, y1) de las componentes conexas de la mascara."""
    lab, n = ndi.label(mascara)
    res = []
    for k, s in enumerate(ndi.find_objects(lab)):
        a = int((lab[s] == k + 1).sum())
        if a > area_min:
            res.append((a, s[1].start, s[1].stop, s[0].start, s[0].stop))
    return res


def base(b):
    return np.array([(b[1] + b[2]) / 2.0, float(b[4])])


def seguir(cands, i0, p0, maxd=75.0):
    """Sigue una mancha desde el frame i0 (en p0): en cada frame, la mas cercana a la prediccion."""
    out = {}
    for sentido in (1, -1):
        prev, vel, last = np.array(p0, float), np.zeros(2), i0
        for i in (range(i0, len(cands)) if sentido == 1 else range(i0, -1, -1)):
            pred = prev + vel * (i - last)
            c = [(np.linalg.norm(base(b) - pred), b) for b in cands[i]]
            c = [x for x in c if x[0] < maxd]
            if c:
                b = min(c, key=lambda x: x[0])[1]
                p = base(b)
                if i != last:
                    vel = (p - prev) / (i - last)
                prev, last = p, i
                out[i] = p
    return out


def main():
    fs = sorted(FRAMES)
    ims = np.array([mpimg.imread(f) for f in fs]).astype(np.float32)
    fondo = np.median(ims, axis=0)
    n = len(fs)
    filas = []

    # --- mujer de buzo rojo ---
    prev = None
    for i in range(n):
        R, G, B = ims[i][..., 0], ims[i][..., 1], ims[i][..., 2]
        rojo = (R > 140) & (R - G > 70) & (R - B > 60)
        rojo[:, :300] = False
        rojo[:, 1100:] = False
        rojo[:330] = False
        rojo[560:] = False
        if prev is not None:
            v = np.zeros_like(rojo)
            v[:, max(prev - 80, 0):prev + 80] = True
            rojo &= v
        ys, xs = np.nonzero(rojo)
        cx = int(np.median(xs[ys < 430])) if np.any(ys < 430) else int(np.median(xs))
        v = np.zeros_like(rojo)
        v[:, cx - 70:cx + 70] = True
        rojo &= v
        ys, xs = np.nonzero(rojo)
        zap = ys > 455
        prev = cx
        if zap.sum() < 5:
            continue
        ymax = np.percentile(ys[zap], 98)
        banda = zap & (ys > ymax - 10)
        filas.append((0, i, xs[banda].mean(), ymax))

    # --- taxi: mancha en movimiento (resta de fondo) ---
    taxi_c = []
    for i in range(n):
        d = np.abs(ims[i] - fondo).sum(axis=2)
        m = d > 90
        m[:150] = False
        m[640:] = False
        m[:, :250] = False
        m[:, 1150:] = False
        m = ndi.binary_closing(ndi.binary_opening(m, iterations=1), iterations=4)
        taxi_c.append([b for b in manchas(m, 150)
                       if b[0] > 1000 and b[4] > 290 and b[1] < 1000 and (b[2] - b[1]) < 330])
    taxi = seguir(taxi_c, 20, (824, 320))
    for i, p in sorted(taxi.items()):
        if 16 <= i <= 43 and i != 26:
            filas.append((1, i, p[0], p[1]))

    # --- camioneta: mancha oscura (resta de fondo no sirve: se mezcla con otros autos) ---
    def forma_ok(b):
        w, h, y1 = b[2] - b[1], b[4] - b[3], b[4]
        return 0.45 * (y1 - 224) < w < 0.95 * (y1 - 224) and 0.25 * w < h < 1.2 * w and b[3] >= 268 and 500 < b[0] < 14000

    prev = None
    for i in range(25, n):
        m = ims[i].max(axis=2) < 75
        m[:268] = False
        m[600:] = False
        m[:, :600] = False
        m[:, 1130:] = False
        m = ndi.binary_closing(ndi.binary_opening(m, iterations=1), iterations=3)
        c = [b for b in manchas(m, 300) if forma_ok(b) and b[1] > 700]
        if not c:
            continue
        if prev is None:
            b = min(c, key=lambda b: abs((b[1] + b[2]) / 2 - 950))
        else:
            b = min(c, key=lambda b: np.linalg.norm(base(b) - prev))
        q = base(b)
        if prev is not None and np.linalg.norm(q - prev) > 90:
            continue
        filas.append((2, i, q[0], q[1]))
        prev = q

    filas.sort()
    with open(SALIDA, "w") as f:
        f.write("# agente (0 mujer, 1 taxi, 2 camioneta); frame (orden alfabetico de los nombres); u; v (pixeles)\n")
        for a, i, u, v in filas:
            f.write("%d %d %.2f %.2f\n" % (a, i, u, v))
    print("escrito", SALIDA, "-", len(filas), "puntos")


if __name__ == "__main__":
    main()
