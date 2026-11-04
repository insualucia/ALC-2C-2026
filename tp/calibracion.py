"""TP - Calibracion de la camara (DLT con puntos 3D)."""
import numpy as np
from alc.basicos import multiplicacionDeMatrices, traspuesta
from alc.lu import res_tri
from alc.svd import svd_reducida


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


def devolverMatricesCalibracion3d(M):
    """
    Descompone la matriz de proyeccion M = K [R | t] (3 x 4) resolviendo las
    incognitas de la ecuacion M = [[au r1 + g r2 + u0 r3 | au t1 + g t2 + u0 t3],
    [av r2 + v0 r3 | av t2 + v0 t3], [r3 | t3]] (ri = filas de R, g = K12).
    M del DLT esta definida salvo escala y signo: se la divide por la norma de
    su tercera fila (que en la ecuacion es la fila r3, de norma 1). Luego se
    despeja de abajo hacia arriba:
      fila 3: r3, t3
      fila 2: v0 = A2.r3,  av = ||A2 - v0 r3||,  r2 = (A2 - v0 r3) / av
      fila 1: u0 = A1.r3,  g = A1.r2,  au = ||A1 - g r2 - u0 r3||,  r1 = ...
      t: se despeja de K t = b por sustitucion hacia atras.
    donde A1, A2, A3 son las filas de la parte 3 x 3 de M y b su ultima columna.
    M: matriz de 3 x 4.
    Retorna R (3 x 3), t (vector de 3) y K (3 x 3, con K33 = 1).
    """
    M = np.asarray(M, dtype=float)

    # escala: la tercera fila de [R | t] tiene norma 1 en su parte de rotacion
    M = M / _norma(M[2, 0:3])
    A1 = M[0, 0:3]
    A2 = M[1, 0:3]
    A3 = M[2, 0:3]
    b = M[:, 3]

    # fila 3
    r3 = A3

    # fila 2: r2 es la componente de A2 ortogonal a r3
    v0 = np.sum(A2 * r3)
    w2 = A2 - v0 * r3
    alfa_v = _norma(w2)
    r2 = w2 / alfa_v

    # fila 1: r1 es la componente de A1 ortogonal a r2 y r3
    u0 = np.sum(A1 * r3)
    gamma = np.sum(A1 * r2)
    w1 = A1 - gamma * r2 - u0 * r3
    alfa_u = _norma(w1)
    r1 = w1 / alfa_u

    K = np.array([[alfa_u, gamma, u0],
                  [0.0, alfa_v, v0],
                  [0.0, 0.0, 1.0]])
    R = np.vstack([r1, r2, r3])
    t = res_tri(K, b, False)

    # R debe ser una rotacion (det = +1); si no, M estaba con el signo opuesto
    # (-M = K (-R) [I | -t]), asi que se cambia el signo de R y t
    if _productoTriple(R[:, 0], R[:, 1], R[:, 2]) < 0:
        R = -R
        t = -t
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
