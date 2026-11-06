"""TP - Deteccion propia de las esquinas internas del tablero de ajedrez (Ej. 3).
Solo usa operaciones permitidas de NumPy (slicing, aritmetica) y el modulo ALC."""
import numpy as np
from alc import *


def _suavizar(I, veces=2, r=2):
    """Suaviza la imagen I (2D) con un filtro de caja de lado 2r+1, aplicado
    `veces` veces (aproxima una gaussiana). Se hace separable, por filas y columnas."""
    I = np.array(I, dtype=float)
    for _ in range(veces):
        A = np.zeros(I.shape)
        for d in range(-r, r + 1):
            A[:, max(0, -d):I.shape[1] - max(0, d)] += I[:, max(0, d):I.shape[1] - max(0, -d)]
        A = A / (2 * r + 1)
        B = np.zeros(I.shape)
        for d in range(-r, r + 1):
            B[max(0, -d):I.shape[0] - max(0, d), :] += A[max(0, d):I.shape[0] - max(0, -d), :]
        I = B / (2 * r + 1)
    return I


def respuestaEsquinas(imagen):
    """
    Respuesta de 'silla de montar' en cada pixel: det(Hessiana) de la imagen
    suavizada, con signo cambiado. Las esquinas internas de un tablero (donde se
    tocan dos cuadrados negros en diagonal) son puntos silla: det(H) muy negativo,
    es decir, respuesta muy positiva.
    imagen: array (alto, ancho, 3) o (alto, ancho) con valores en [0, 1].
    Retorna un array (alto, ancho) con la respuesta (0 en el borde).
    """
    I = np.array(imagen, dtype=float)
    if I.ndim == 3:
        I = (I[:, :, 0] + I[:, :, 1] + I[:, :, 2]) / 3
    I = _suavizar(I)
    Ixx = np.zeros(I.shape)
    Iyy = np.zeros(I.shape)
    Ixy = np.zeros(I.shape)
    Ixx[:, 1:-1] = I[:, 2:] - 2 * I[:, 1:-1] + I[:, :-2]
    Iyy[1:-1, :] = I[2:, :] - 2 * I[1:-1, :] + I[:-2, :]
    Ixy[1:-1, 1:-1] = (I[2:, 2:] - I[2:, :-2] - I[:-2, 2:] + I[:-2, :-2]) / 4
    return -(Ixx * Iyy - Ixy * Ixy)


def candidatosEsquinas(R, cantidad=120, radio=12, umbral=0.15):
    """
    Maximos locales de la respuesta R, con supresion de no-maximos.
    Se toma repetidamente el maximo, se lo refina con el centroide ponderado de
    R en un entorno de 3 pixeles (precision subpixel) y se anula R a su alrededor.
    Retorna array (k, 2) de (u, v) con k <= cantidad.
    """
    R = np.array(R, dtype=float)
    # el borde de la imagen da falsas respuestas: se anula
    R[:10, :] = 0; R[-10:, :] = 0; R[:, :10] = 0; R[:, -10:] = 0
    tope = np.max(R)
    pts = []
    for _ in range(cantidad):
        m = np.max(R)
        if m < umbral * tope:
            break
        v, u = np.where(R == m)
        v = v[0]
        u = u[0]
        v0 = max(v - 3, 0); v1 = min(v + 4, R.shape[0])
        u0 = max(u - 3, 0); u1 = min(u + 4, R.shape[1])
        W = np.maximum(R[v0:v1, u0:u1], 0)
        gu = np.arange(u0, u1) + 0.0
        gv = np.arange(v0, v1) + 0.0
        sw = np.sum(W)
        pts.append([np.sum(W * gu.reshape(1, -1)) / sw, np.sum(W * gv.reshape(-1, 1)) / sw])
        R[max(v - radio, 0):v + radio + 1, max(u - radio, 0):u + radio + 1] = 0
    return np.array(pts)


def _masCercano(pts, x, excluir=None):
    """Indice del punto de pts (k x 2) mas cercano a x, y la distancia."""
    d = np.sqrt(np.sum((pts - x.reshape(1, 2)) ** 2, axis=1))
    if excluir is not None:
        d = d + 1e9 * excluir
    i = 0
    for k in range(len(d)):
        if d[k] < d[i]:
            i = k
    return i, d[i]


def ordenarTablero(pts, nu=11, nv=7):
    """
    Dado un conjunto de candidatos a esquina (que incluye tambien esquinas
    externas del tablero), encuentra el bloque de nu x nv esquinas internas y las
    devuelve ordenadas: indice k = j * nu + i, con i = 0..nu-1 hacia la derecha
    (eje U) y j = 0..nv-1 hacia abajo (eje V).
    Idea: se etiqueta cada candidato con coordenadas enteras (i, j) de la grilla
    creciendo desde el punto central (siempre interno), prediciendo cada vecino
    con el paso local; luego se busca la ventana nu x nv completamente ocupada.
    Retorna array (nu*nv, 2), o None si no se pudo ordenar.
    """
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    centro = np.array([np.sum(pts[:, 0]) / n, np.sum(pts[:, 1]) / n])
    sem, _ = _masCercano(pts, centro)
    # paso de la grilla: distancia al vecino mas cercano del semillero
    usados = np.zeros(n)
    usados[sem] = 1
    v1, paso = _masCercano(pts, pts[sem], usados)
    # los 4 vecinos de la grilla: puntos a distancia ~paso del semillero
    vec = []
    for k in range(n):
        d = np.sqrt(np.sum((pts[k] - pts[sem]) ** 2))
        if k != sem and d < 1.3 * paso:
            vec.append(pts[k] - pts[sem])
    # direccion "derecha": mayor componente x con |y| chico; "abajo" analogo
    a1 = None
    a2 = None
    for w in vec:
        if abs(w[0]) > abs(w[1]) and w[0] > 0:
            a1 = w
        if abs(w[1]) > abs(w[0]) and w[1] > 0:
            a2 = w
    if a1 is None or a2 is None:
        return None
    # crecimiento: etiquetas[(i, j)] = indice de punto
    etiq = {(0, 0): sem}
    pos = {(0, 0): pts[sem]}
    cola = [(0, 0)]
    while len(cola) > 0:
        i, j = cola.pop(0)
        for di, dj, base in [(1, 0, a1), (-1, 0, -a1), (0, 1, a2), (0, -1, -a2)]:
            q = (i + di, j + dj)
            if q in etiq:
                continue
            # paso local: si el vecino opuesto ya esta, se usa su diferencia
            prev = (i - di, j - dj)
            if prev in pos:
                salto = pos[(i, j)] - pos[prev]
            else:
                salto = base
            pred = pos[(i, j)] + salto
            k, d = _masCercano(pts, pred)
            ya = False
            for kk in etiq.values():
                if kk == k:
                    ya = True
            if d < 0.3 * paso and not ya:
                etiq[q] = k
                pos[q] = pts[k]
                cola.append(q)
    # rango de etiquetas: incluye el anillo exterior, asi que el bloque interno
    # es la ventana nu x nv completamente ocupada y CENTRADA en ese rango
    imin = 10 ** 6; imax = -10 ** 6; jmin = 10 ** 6; jmax = -10 ** 6
    for (i, j) in etiq:
        imin = min(imin, i); imax = max(imax, i)
        jmin = min(jmin, j); jmax = max(jmax, j)
    mejor = None
    mejorScore = 10 ** 9
    for i0 in range(imin, imax + 1):
        for j0 in range(jmin, jmax + 1):
            ok = True
            for j in range(nv):
                for i in range(nu):
                    if (i0 + i, j0 + j) not in etiq:
                        ok = False
            if ok:
                score = abs(2 * i0 + nu - 1 - imin - imax) + abs(2 * j0 + nv - 1 - jmin - jmax)
                if score < mejorScore:
                    mejorScore = score
                    mejor = (i0, j0)
    if mejor is None:
        return None
    i0, j0 = mejor
    out = np.zeros((nu * nv, 2))
    for j in range(nv):
        for i in range(nu):
            out[j * nu + i] = pts[etiq[(i0 + i, j0 + j)]]
    return out


def esquinasTablero(imagen, nu=11, nv=7):
    """Esquinas internas (nu*nv x 2) de una imagen de tablero, ordenadas, o None."""
    return ordenarTablero(candidatosEsquinas(respuestaEsquinas(imagen)), nu, nv)
