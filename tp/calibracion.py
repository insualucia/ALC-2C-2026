"""TP - Calibracion de la camara (DLT con puntos 3D)."""
import numpy as np
from alc import *


def obtenerQ3d(imgPts, objPts):
    """
    Arma la matriz Q (2n x 12) del sistema Qm = 0 (ecuacion 6).
    imgPts: array de n x 2 con los (u, v) de cada punto en la imagen.
    objPts: array de n x 3 con los (U, V, W) de cada punto en el mundo.
    Retorna Q, donde cada punto aporta dos filas:
        [P^T, 0, -u P^T]  y  [0, P^T, -v P^T],  con P = (U, V, W, 1).
    """
    imgPts = np.asarray(imgPts, dtype=float)
    objPts = np.asarray(objPts, dtype=float)
    n = objPts.shape[0]
    Q = np.zeros((2 * n, 12))
    for i in range(n):
        u = imgPts[i, 0]
        v = imgPts[i, 1]
        P = np.array([objPts[i, 0], objPts[i, 1], objPts[i, 2], 1.0])
        Q[2 * i, 0:4] = P
        Q[2 * i, 8:12] = -u * P
        Q[2 * i + 1, 4:8] = P
        Q[2 * i + 1, 8:12] = -v * P
    return Q


def resolverDLT3D(Q):
    """
    Resuelve  min ||Qm||^2  sujeto a ||m|| = 1  (ecuacion 8).
    La solucion es el vector singular derecho de Q asociado al menor valor
    singular (el autovector de Q^T Q de menor autovalor).
    Q: matriz de 2n x 12.
    Retorna M, la matriz de proyeccion de 3 x 4 (m reordenado por filas).
    """
    Q = np.asarray(Q, dtype=float)
    # k = 12: hay que conservar tambien el menor valor singular, que es el
    # que buscamos (con k="max" se lo descartaria por ser casi nulo)
    _, S, V = svd_reducida(Q, k=Q.shape[1])
    m = V[:, -1]
    M = np.zeros((3, 4))
    for j in range(3):
        M[j, :] = m[4 * j:4 * j + 4]
    return M


def _productoTriple(r1, r2, r3):
    """Determinante de la matriz con columnas r1, r2, r3: r1 . (r2 x r3)."""
    c = np.array([r2[1] * r3[2] - r2[2] * r3[1],
                  r2[2] * r3[0] - r2[0] * r3[2],
                  r2[0] * r3[1] - r2[1] * r3[0]])
    return np.sum(r1 * c)


def _norma(v):
    """Norma euclidea de un vector."""
    return np.sqrt(np.sum(v * v))


def devolverMatricesCalibracion3d(M, imponerAlfasIguales=False, imponerGammaCero=False):

    """
    Descompone la matriz de proyección M = K [R t] en sus parámetros
    intrínsecos (K) y extrínsecos (R, t), imponiendo que R sea ortonormal.

    Parámetros:
        M : array (3, 4), matriz de proyección (a menos de escala), por
            ejemplo la que devuelve resolverDLT3D.
        imponerAlfasIguales : bool, opcional. Si es True, fuerza alfa_u = alfa_v
            (píxeles cuadrados), reemplazando ambos por su promedio.
        imponerGammaCero : bool, opcional. Si es True, fuerza K12 = gamma = 0
            (ejes del sensor perpendiculares).

    Devuelve:
        R : array (3, 3), matriz de rotación (ortonormal).
        t : array (3,), vector de traslación, en las unidades del mundo.
        K : array (3, 3), matriz de parámetros intrínsecos, con K33 = 1.
    """

    M = np.array(M, dtype=float)

    # --- 1. Sacar la escala desconocida: dividir M por la norma de m3 ---
    # Las tres primeras entradas de la fila 3 son c * r3, y r3 tiene norma 1.
    M = M / norma(M[2, 0:3], 2)

    # --- 2. Elegir el signo: la mira tiene que estar delante de la cámara ---
    # t3 = M[2, 3] tiene que ser positivo; si no, cambiamos el signo de M.
    if M[2, 3] < 0:
        M = -M

    # Bloque izquierdo (K R): m1, m2, m3 son las tres primeras entradas de cada fila
    m1 = M[0, 0:3]
    m2 = M[1, 0:3]
    m3 = M[2, 0:3]

    # --- 3. Despeje de abajo hacia arriba usando la ortonormalidad de R ---
    # Fila 3: m3 = r3
    r3 = m3

    # Fila 2: m2 = alfa_v r2 + v0 r3
    v0 = productoInterno(m2, r3)                    # r3 . m2 = v0
    alfaV = norma(m2 - v0 * r3, 2)                  # ||m2 - v0 r3|| = alfa_v
    r2 = (m2 - v0 * r3) / alfaV

    # Fila 1: m1 = alfa_u r1 + gamma r2 + u0 r3
    u0 = productoInterno(m1, r3)                    # r3 . m1 = u0
    gamma = productoInterno(m1, r2)                 # r2 . m1 = gamma
    alfaU = norma(m1 - gamma * r2 - u0 * r3, 2)     # ||m1 - gamma r2 - u0 r3|| = alfa_u
    r1 = (m1 - gamma * r2 - u0 * r3) / alfaU

    # Armamos R con r1, r2, r3 como filas
    R = np.vstack([r1, r2, r3])

    # --- 4. Restricciones opcionales sobre K (Ejercicio 2d) ---
    if imponerGammaCero:
        gamma = 0.0
    if imponerAlfasIguales:
        alfaPromedio = (alfaU + alfaV) / 2
        alfaU = alfaPromedio
        alfaV = alfaPromedio

    K = np.array([[alfaU, gamma, u0],
                  [0.0,   alfaV, v0],
                  [0.0,   0.0,   1.0]])

    # --- 5. Traslación: la última columna de M es K t (sistema triangular) ---
    # Se resuelve por sustitución hacia atrás, con la K ya definida.
    m14 = M[0, 3]
    m24 = M[1, 3]
    m34 = M[2, 3]
    t3 = m34
    t2 = (m24 - v0 * t3) / alfaV
    t1 = (m14 - gamma * t2 - u0 * t3) / alfaU
    t = np.array([t1, t2, t3])

    return R, t, K


def verificarCalibracion3d(M, R, t, K, imgPts, objPts, tol=1e-8):
    """
    Chequeos de la calibracion obtenida con devolverMatricesCalibracion3d.
    M: matriz de proyeccion (3 x 4); R, t, K: resultado de la descomposicion.
    imgPts (n x 2) y objPts (n x 3): puntos usados para calibrar.
    Retorna un diccionario con:
      ortogonalidad: max |R R^T - I|          (debe ser ~ 0)
      det_R:         determinante de R        (debe ser +1)
      K_triangular:  max |K bajo la diagonal| (debe ser ~ 0), K33 = 1
      dif_M:         max |K [R|t] - M/escala| (debe ser ~ 0)
      error_reproyeccion: error maximo en pixeles al reproyectar objPts
      delante_camara: True si todos los puntos tienen Z > 0 en la camara
    """
    imgPts = np.asarray(imgPts, dtype=float)
    objPts = np.asarray(objPts, dtype=float)
    n = objPts.shape[0]

    RRt = multiplicacionDeMatrices(R, traspuesta(R))
    ortogonalidad = np.max(np.abs(RRt - np.eye(3)))
    det_R = _productoTriple(R[:, 0], R[:, 1], R[:, 2])
    K_triangular = max(abs(K[1, 0]), abs(K[2, 0]), abs(K[2, 1]))

    # K [R | t] debe coincidir con M salvo un factor de escala (global)
    Rt = np.hstack([R, t.reshape(3, 1)])
    KRt = multiplicacionDeMatrices(K, Rt)
    escala = np.sum(M * KRt) / np.sum(KRt * KRt)
    dif_M = np.max(np.abs(M - escala * KRt))

    # reproyeccion y profundidad en el sistema de la camara
    P = np.hstack([objPts, np.ones((n, 1))])
    proy = multiplicacionDeMatrices(P, traspuesta(multiplicacionDeMatrices(K, Rt)))
    uv = proy[:, 0:2] / proy[:, 2:3]
    error_reproyeccion = np.max(np.abs(uv - imgPts))
    Zcam = multiplicacionDeMatrices(P, traspuesta(Rt))[:, 2]
    delante_camara = bool(np.all(Zcam > 0))

    return {"ortogonalidad": ortogonalidad, "det_R": det_R,
            "K_triangular": K_triangular, "dif_M": dif_M,
            "error_reproyeccion": error_reproyeccion,
            "delante_camara": delante_camara}
