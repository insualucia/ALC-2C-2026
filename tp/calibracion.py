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
    La solucion es el autovector de Q^T Q de menor autovalor (el vector singular
    derecho de Q asociado al menor valor singular).
    Q: matriz de 2n x 12.
    Retorna M, la matriz de proyeccion de 3 x 4 (m reordenado por filas).
    """
    Q = np.asarray(Q, dtype=float)
    # autovector de menor autovalor de Q^T Q (ver _vectorNucleo): no se usa
    # svd_reducida porque descarta los valores singulares casi nulos
    m = _vectorNucleo(Q)
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

def obtenerQ2d(imgPts, objPts):
    """
    Arma la matriz Q (2n x 9) del sistema Qh = 0 (ecuacion 11), para puntos coplanares.
    imgPts: array de n x 2 con los (u, v) de cada punto en la imagen.
    objPts: array de n x 2 con los (U, V) de cada punto en el plano del objeto (W = 0).
            Si viene de n x 3, la tercera columna debe ser 0 y se ignora.
    Retorna Q, donde cada punto aporta dos filas:
        [P^T, 0, -u P^T]  y  [0, P^T, -v P^T],  con P = (U, V, 1).
    """
    imgPts = np.asarray(imgPts, dtype=float)
    objPts = np.asarray(objPts, dtype=float)

    # Si vienen coordenadas 3D, verificar que sean coplanares en W = 0
    if objPts.shape[1] == 3 and np.any(np.abs(objPts[:, 2]) > 0):
        raise ValueError("obtenerQ2d: los puntos deben tener W = 0 (plano del objeto).")

    n = objPts.shape[0]
    Q = np.zeros((2 * n, 9))
    for i in range(n):
        u = imgPts[i, 0]
        v = imgPts[i, 1]
        # Solo U y V: la columna de W desaparece porque W = 0
        P = np.array([objPts[i, 0], objPts[i, 1], 1.0])
        # Fila de u: h1^T P - u * h3^T P = 0
        Q[2 * i, 0:3] = P
        Q[2 * i, 6:9] = -u * P
        # Fila de v: h2^T P - v * h3^T P = 0
        Q[2 * i + 1, 3:6] = P
        Q[2 * i + 1, 6:9] = -v * P
    return Q



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



def _vectorNucleo(M):
    """
    Vector unitario x que minimiza ||M x|| (a menos de signo): el autovector de
    M^T M de menor autovalor, calculado con diagRH del modulo ALC.
    Se usa en lugar de svd_reducida porque esta descarta los valores singulares
    (casi) nulos, y con datos exactos el menor valor singular es justo ~0: la
    ultima columna de V ya no seria el vector del nucleo.
    M: matriz de m x n. Retorna un vector de n componentes.
    """
    M = np.asarray(M, dtype=float)
    S, D = diagRH(multiplicacionDeMatrices(traspuesta(M), M))
    n = M.shape[1]
    imin = 0
    for j in range(n):
        if D[j, j] < D[imin, imin]:
            imin = j
    return np.array(S)[:, imin]


def _normalizacionIsotropica(pts):
    """
    Normalizacion de Hartley (seccion 2.3): matriz T (3 x 3) tal que los puntos
    T [x y 1]^T tienen centroide en el origen y distancia media al origen sqrt(2).
    pts: array de n x 2. Retorna T.
    """
    pts = np.asarray(pts, dtype=float)
    n = pts.shape[0]
    cx = np.sum(pts[:, 0]) / n
    cy = np.sum(pts[:, 1]) / n
    distMedia = np.sum(np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)) / n
    s = np.sqrt(2) / distMedia
    return np.array([[s, 0.0, -s * cx],
                     [0.0, s, -s * cy],
                     [0.0, 0.0, 1.0]])


def _aplicarT(T, pts):
    """Aplica la transformacion homogenea T (3 x 3) a puntos 2D (n x 2)."""
    pts = np.asarray(pts, dtype=float)
    n = pts.shape[0]
    P = np.hstack([pts, np.ones((n, 1))])
    Y = multiplicacionDeMatrices(P, traspuesta(T))
    return Y[:, 0:2] / Y[:, 2:3]


def resolverH(imgPts, objPts, normalizar=True):
    """
    Estima la homografia H (3 x 3) del plano del tablero a la imagen, con
    lambda (u, v, 1)^T = H (U, V, 1)^T, resolviendo  min ||Qh||^2 s.a. ||h|| = 1.
    imgPts: n x 2 (u, v) en pixeles. objPts: n x 2 (U, V) en el plano (o n x 3 con W = 0).
    normalizar: si es True usa la normalizacion isotropica de Hartley (ec. 14):
        se resuelve para los puntos normalizados y luego H = T_img^-1 H_norm T_world.
    Retorna H (3 x 3) con norma de Frobenius 1.
    """
    imgPts = np.asarray(imgPts, dtype=float)
    objPts = np.asarray(objPts, dtype=float)[:, 0:2]
    if normalizar:
        Timg = _normalizacionIsotropica(imgPts)
        Tworld = _normalizacionIsotropica(objPts)
        imgN = _aplicarT(Timg, imgPts)
        objN = _aplicarT(Tworld, objPts)
    else:
        imgN = imgPts
        objN = objPts
    Q = obtenerQ2d(imgN, objN)
    h = _vectorNucleo(Q)
    H = np.zeros((3, 3))
    for j in range(3):
        H[j, :] = h[3 * j:3 * j + 3]
    if normalizar:
        H = multiplicacionDeMatrices(multiplicacionDeMatrices(inversa(Timg), H), Tworld)
    return H / np.sqrt(np.sum(H * H))


def _vij(H, i, j):
    """
    Vector v_ij (6,) con  c_i^T B c_j = v_ij^T b,  donde c_i es la columna i de H
    y b = (B11, B12, B22, B13, B23, B33). (Zhang, seccion 3.)
    """
    hi = H[:, i]
    hj = H[:, j]
    return np.array([hi[0] * hj[0],
                     hi[0] * hj[1] + hi[1] * hj[0],
                     hi[1] * hj[1],
                     hi[2] * hj[0] + hi[0] * hj[2],
                     hi[2] * hj[1] + hi[1] * hj[2],
                     hi[2] * hj[2]])


def armarA(Hlst):
    """
    Arma la matriz A (2N x 6) del sistema A b = 0 (ecuacion 13), con N homografias.
    Hlst: lista de N matrices H_j de 3 x 3 (una por vista).
    Cada vista aporta dos filas:  v12^T b = 0  y  (v11 - v22)^T b = 0,
    que salen de r1^T r2 = 0 y ||r1|| = ||r2||. Se normaliza cada H a norma 1
    para que todas las vistas pesen parecido.
    """
    N = len(Hlst)
    A = np.zeros((2 * N, 6))
    for k in range(N):
        H = np.asarray(Hlst[k], dtype=float)
        H = H / np.sqrt(np.sum(H * H))
        A[2 * k, :] = _vij(H, 0, 1)
        A[2 * k + 1, :] = _vij(H, 0, 0) - _vij(H, 1, 1)
    return A


def _cholesky(B):
    """
    Factorizacion de Cholesky B = L L^T (L triangular inferior) de una matriz
    simetrica definida positiva, a partir de la factorizacion LDV del modulo ALC:
    para B simetrica B = L1 D L1^T, y si D > 0, L = L1 D^(1/2).
    Lanza ValueError si algun elemento de D es <= 0 (B no es definida positiva).
    """
    n = B.shape[0]
    L1, D, V = calculaLDV(B)
    L = np.zeros((n, n))
    for j in range(n):
        if D[j, j] <= 0:
            raise ValueError("_cholesky: la matriz no es definida positiva.")
        L[:, j] = L1[:, j] * np.sqrt(D[j, j])
    return L


def calcularMatrizK2D(A):
    """
    Calcula la matriz de intrinsecos K (3 x 3) a partir de A (2N x 6).
    1) b = vector singular derecho de menor valor singular de A (A b = 0, ||b|| = 1).
    2) b es definido salvo signo: se elige el que deja a B = K^-T K^-1 definida
       positiva (B11 > 0 y Cholesky exitoso).
    3) B = L L^T (Cholesky) => L = K^-T => K = (L^T)^-1, normalizada con K33 = 1.
    Se necesitan N >= 3 vistas (N >= 2 si se supone gamma = 0), con poses no degeneradas.
    """
    A = np.asarray(A, dtype=float)
    b = _vectorNucleo(A)
    if b[0] < 0:
        b = -b
    B = np.array([[b[0], b[1], b[3]],
                  [b[1], b[2], b[4]],
                  [b[3], b[4], b[5]]])
    L = _cholesky(B)
    K = inversa(traspuesta(L))
    return K / K[2, 2]


def calcularKRestringida(Hlst, centro):
    """
    Intrinsecos con restricciones (Ej. 5.a): gamma = 0, pixeles cuadrados
    (alfa_u = alfa_v = f) y punto principal conocido (cu, cv). Queda una sola
    incognita, f.

    Con esa K, B = K^-T K^-1 = (1/f^2) [[1, 0, -cu], [0, 1, -cv], [-cu, -cv, cu^2+cv^2+f^2]],
    y su vector b = (B11, B12, B22, B13, B23, B33) resulta  b = x p + e, con x = 1/f^2,
    p = (1, 0, 1, -cu, -cv, cu^2+cv^2) y e = (0, 0, 0, 0, 0, 1).
    El sistema A b = 0 (de armarA) se vuelve  x (A p) = -(A e): un cuadrado minimo
    de una sola incognita,  x = -(Ap . Ae) / (Ap . Ap).

    Hlst: lista de N homografias (una por vista); basta N >= 1.
    centro: (cu, cv) en pixeles.
    Retorna K (3 x 3) y el residuo relativo ||x A p + A e|| / ||A e|| (0 si las H
    son compatibles con esa K; crece con el ruido).
    Lanza ValueError si x <= 0 (no hay f real).
    """
    cu, cv = float(centro[0]), float(centro[1])
    A = armarA(Hlst)
    p = np.array([1.0, 0.0, 1.0, -cu, -cv, cu * cu + cv * cv])
    e = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    Ap = calcularAx(A, p)
    Ae = calcularAx(A, e)
    x = -np.sum(Ap * Ae) / np.sum(Ap * Ap)
    if x <= 0:
        raise ValueError("calcularKRestringida: x = 1/f^2 <= 0, no hay f real.")
    f = 1.0 / np.sqrt(x)
    residuo = np.sqrt(np.sum((x * Ap + Ae) ** 2)) / np.sqrt(np.sum(Ae * Ae))
    K = np.array([[f, 0.0, cu],
                  [0.0, f, cv],
                  [0.0, 0.0, 1.0]])
    return K, residuo


def reproyectarPose(K, R, t, ptsMundo):
    """
    Proyecta puntos del mundo a la imagen con la pose (R, t) y los intrinsecos K
    (Ej. 5.b):  x = K (R P + t),  (u, v) = (x1 / x3, x2 / x3).
    ptsMundo: n x 3 (X, Y, Z) en metros (o n x 2, con Z = 0). Retorna n x 2 en pixeles.
    """
    P = np.asarray(ptsMundo, dtype=float)
    if P.shape[1] == 2:
        P = np.hstack([P, np.zeros((P.shape[0], 1))])
    n = P.shape[0]
    uv = np.zeros((n, 2))
    for i in range(n):
        X = calcularAx(R, P[i]) + t
        x = calcularAx(K, X)
        uv[i, 0] = x[0] / x[2]
        uv[i, 1] = x[1] / x[2]
    return uv


def pendienteRecta(t, y):
    """
    Pendiente de la recta de minimos cuadrados y = a + b t:  b = sum((t - tm)(y - ym)) / sum((t - tm)^2).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    tm = np.mean(t)
    return np.sum((t - tm) * (y - np.mean(y))) / np.sum((t - tm) ** 2)


def velocidadMedia(t, P):
    """
    Velocidad media de un agente que se mueve en el piso (Ej. 5.c).
    t: n instantes en segundos. P: n x 2 posiciones (X, Y) en metros.
    Ajusta una recta por minimos cuadrados a X(t) y a Y(t): la pendiente es la velocidad
    en cada eje. Retorna (v, (vx, vy)), con v = sqrt(vx^2 + vy^2) en m/s.
    """
    P = np.asarray(P, dtype=float)
    vx = pendienteRecta(t, P[:, 0])
    vy = pendienteRecta(t, P[:, 1])
    return np.sqrt(vx * vx + vy * vy), (vx, vy)


def velocidadesLocales(t, P, k=3):
    """
    Velocidad en cada instante a partir de las 2k + 1 posiciones vecinas (ventana de k
    puntos a cada lado, mas chica en los bordes, minimo 3 puntos): pendiente de la recta
    de minimos cuadrados de X(t) e Y(t) en la ventana. Mas estable que diferencias de a
    pares, porque el error de medir un pixel (unos 3 px) pesa mucho sobre dt de 0,03 a 0,25 s.
    Retorna v (n,) en m/s.
    """
    t = np.asarray(t, dtype=float)
    P = np.asarray(P, dtype=float)
    n = len(t)
    v = np.zeros(n)
    for i in range(n):
        a = max(0, i - k)
        b = min(n, i + k + 1)
        if b - a < 3:
            a, b = max(0, b - 3), min(n, a + 3)
        vx = pendienteRecta(t[a:b], P[a:b, 0])
        vy = pendienteRecta(t[a:b], P[a:b, 1])
        v[i] = np.sqrt(vx * vx + vy * vy)
    return v


def obtenerPose2D(Hj, K):
    """
    Pose (R_j, t_j) de la vista j a partir de su homografia H_j y de K (ec. 12 y h).
    H_j ~ K [r1 r2 t]  =>  r1 = lam K^-1 c1,  r2 = lam K^-1 c2,  r3 = r1 x r2,
    t = lam K^-1 c3,  con lam = 1 / ||K^-1 c1||.
    El signo de lam se elige para que t3 > 0 (el tablero queda delante de la camara).
    Como el ruido hace que [r1 r2 r3] no sea exactamente una rotacion, se la
    proyecta a la rotacion ortogonal mas cercana (R = U V^T de su SVD), calculada
    con iteracion de Newton.
    Retorna R (3 x 3) y t (3,) en las unidades del tablero (mm).
    """
    Hj = np.asarray(Hj, dtype=float)
    Kinv = inversa(K)
    c1 = Hj[:, 0]
    c2 = Hj[:, 1]
    c3 = Hj[:, 2]
    k1 = calcularAx(Kinv, c1)
    k2 = calcularAx(Kinv, c2)
    k3 = calcularAx(Kinv, c3)
    lam = 1.0 / norma(k1, 2)
    if lam * k3[2] < 0:
        lam = -lam
    r1 = lam * k1
    r2 = lam * k2
    r3 = np.array([r1[1] * r2[2] - r1[2] * r2[1],
                   r1[2] * r2[0] - r1[0] * r2[2],
                   r1[0] * r2[1] - r1[1] * r2[0]])
    t = lam * k3
    R = np.array([r1, r2, r3]).T          # r1, r2, r3 como columnas
    # Proyeccion a la rotacion mas cercana (el factor U V^T de la SVD de R) por
    # iteracion de Newton R <- (R + R^-T) / 2: converge en 2-3 pasos a la misma
    # matriz y queda ortonormal a precision de maquina. (Con svd_reducida de ALC
    # quedaba ortonormal solo hasta ~3e-4: R ya es casi rotacion, asi que R^T R ~ I
    # tiene los 3 autovalores casi iguales y sus autovectores salen mal separados.)
    for _ in range(50):
        Rnueva = (R + traspuesta(inversa(R))) / 2
        cambio = np.max(np.abs(Rnueva - R))
        R = Rnueva
        if cambio < 1e-15:
            break
    return R, t


# ---------------------------------------------------------------------------
# Ejercicio 4 - Homografia del plano de la cancha
# ---------------------------------------------------------------------------

def proyectarPuntos(H, pts):
    """
    Aplica la homografia H (3 x 3) a puntos 2D del plano: x = H (p, 1)^T y luego
    se divide por la tercera componente (eso es lo que modela la perspectiva).
    H: cancha -> imagen (la que devuelve resolverH). Con inversa(H) se pasa de
    pixeles de la foto a coordenadas de la cancha (metros).
    pts: array de n x 2 (o un solo punto de 2 componentes).
    Retorna un array de n x 2.
    """
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    n = pts.shape[0]
    P = np.hstack([pts, np.ones((n, 1))])
    X = multiplicacionDeMatrices(P, traspuesta(H))
    return X[:, 0:2] / X[:, 2:3]


def circuloEnPlano(centro, radio, n=100):
    """
    Puntos de una circunferencia de centro (U, V) y radio dados, en el plano.
    Sin usar seno ni coseno: se usa la parametrizacion racional
        x = (1 - t^2) / (1 + t^2),  y = 2 t / (1 + t^2),  t en [-1, 1],
    que recorre el semicirculo de la derecha (angulos de -90 a 90 grados con
    paso casi uniforme); el de la izquierda es su reflejo en x.
    centro: (U, V). radio: en las mismas unidades (metros). n: puntos por semicirculo.
    Retorna un array de (2 n + 1) x 2 que arranca y termina en el mismo punto.
    """
    t = np.arange(n + 1) / n * 2 - 1
    x = (1 - t * t) / (1 + t * t)
    y = 2 * t / (1 + t * t)
    # derecha de abajo hacia arriba, luego izquierda de arriba hacia abajo
    xs = np.hstack([x, -x[::-1][1:]])
    ys = np.hstack([y, y[::-1][1:]])
    return np.hstack([(centro[0] + radio * xs).reshape(-1, 1),
                      (centro[1] + radio * ys).reshape(-1, 1)])


# ---------------------------------------------------------------------------
# Ejercicio 6 - POSIT coplanar (aterrizaje suave)
# ---------------------------------------------------------------------------

def _cruz(a, b):
    """Producto vectorial de dos vectores de 3 componentes."""
    return np.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def calculoLambdaMu(I0, J0):
    """
    Escalares (lambda, mu) de la ecuacion 16, tales que I = I0 + lambda n y J = J0 + mu n
    cumplen I.J = 0 y ||I|| = ||J|| (n: normal unitaria al plano del objeto).
    Con C = lambda + i mu:  C^2 = (||J0||^2 - ||I0||^2) - 2i (I0 . J0)  (ec. 17).
    Si C^2 = a + i b, sus raices son +-C con
        lambda = sqrt((|C^2| + a) / 2),  mu = sign(b) sqrt((|C^2| - a) / 2)
    (el signo de mu sale de lambda mu = b / 2 = -I0 . J0).
    Retorna la raiz de lambda >= 0; la otra solucion es (-lambda, -mu).
    """
    I0 = np.asarray(I0, dtype=float)
    J0 = np.asarray(J0, dtype=float)
    a = np.sum(J0 * J0) - np.sum(I0 * I0)
    b = -2.0 * np.sum(I0 * J0)
    r = np.sqrt(a * a + b * b)
    lam = np.sqrt(max((r + a) / 2.0, 0.0))
    mu = np.sqrt(max((r - a) / 2.0, 0.0))
    if b < 0:
        mu = -mu
    return lam, mu


def calcularSalidaPOSIT(Is, Js, fx, m0, M0):
    """
    Pose (i, j, k, Z0, R, t) a partir de una solucion (I, J) del paso POS.
    Is, Js: vectores I y J (3,). fx: focal en pixeles. m0: imagen (x0, y0) de M0,
    ya medida respecto del centro principal. M0: coordenadas de M0 en el sistema del
    objeto (los a_i del paso POS son relativos a M0).
    Z0 = f / ||I|| (se usa el promedio de ||I|| y ||J||), i = I/||I||, j = J/||J||,
    k = i x j, R = [i; j; k] (filas) y T = (Z0/f)(x0, y0, f) es la posicion de M0 en la
    camara. Como X = R (P - M0) + T = R P + t, resulta t = T - R M0.
    """
    Is = np.asarray(Is, dtype=float)
    Js = np.asarray(Js, dtype=float)
    nI = norma(Is, 2)
    nJ = norma(Js, 2)
    Z0 = fx / ((nI + nJ) / 2.0)
    i = Is / nI
    j = Js / nJ
    k = _cruz(i, j)
    R = np.array([i, j, k])
    T = (Z0 / fx) * np.array([m0[0], m0[1], fx])
    t = T - calcularAx(R, np.asarray(M0, dtype=float))
    return i, j, k, Z0, R, t


def evaluarPose(R, t, obj_vecs, img_vecs, K):
    """
    Calidad de una pose. Proyecta obj_vecs (n x 3) con (R, t) y K; un punto es valido
    si queda delante de la camara (Z > 0). Retorna (cantidad de validos, RMS en pixeles
    contra img_vecs (n x 2, pixeles), calculado sobre los validos; inf si no hay ninguno).
    """
    obj_vecs = np.asarray(obj_vecs, dtype=float)
    img_vecs = np.asarray(img_vecs, dtype=float)
    n = obj_vecs.shape[0]
    validos = 0
    suma = 0.0
    for p in range(n):
        X = calcularAx(R, obj_vecs[p]) + t
        if X[2] <= 0:
            continue
        x = calcularAx(K, X)
        du = x[0] / x[2] - img_vecs[p, 0]
        dv = x[1] / x[2] - img_vecs[p, 1]
        suma += du * du + dv * dv
        validos += 1
    if validos == 0:
        return 0, np.inf
    return validos, np.sqrt(suma / validos)


def _pasoPOS(A, b):
    """Solucion de minimos cuadrados de A x = b con la pseudoinversa (SVD): x = V S^-1 U^T b."""
    U, S, V = svd_reducida(A)
    c = calcularAx(traspuesta(U), b) / S
    return calcularAx(V, c)


def aplicarPOSIT(obj_vecs, img_vecs, K, MaxIt=50, atol=1e-6):
    """
    POSIT coplanar (seccion 4). obj_vecs: n x 3 puntos del objeto, el primero es el
    de referencia M0 (n >= 4, coplanares). img_vecs: n x 2 sus pixeles (u, v).
    K: intrinsecos (se usan fx, fy, cu, cv). Se itera hasta MaxIt veces o hasta que el
    RMS cambie menos que atol entre dos iteraciones.
    En cada iteracion:  eps_i fijos -> I0 = A+ b_u, J0 = A+ b_v -> n (nucleo de A) ->
    dos soluciones (lambda, mu) y (-lambda, -mu) -> pose de cada una -> se descartan las
    que dejan algun punto detras de la camara (Z <= 0) -> se queda la de menor RMS ->
    eps_i = a_i . k / Z0.
    Retorna R, t (X_camara = R P + t). Lanza ValueError si ninguna solucion es valida.
    """
    obj_vecs = np.asarray(obj_vecs, dtype=float)
    img_vecs = np.asarray(img_vecs, dtype=float)
    n = obj_vecs.shape[0]
    fx, fy, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # coordenadas centradas; y se reescala para que la focal sea fx en los dos ejes
    x = img_vecs[:, 0] - cu
    y = (img_vecs[:, 1] - cv) * fx / fy
    M0 = obj_vecs[0]
    A = obj_vecs[1:] - M0
    nu = _vectorNucleo(A)
    eps = np.zeros(n - 1)
    R, t, rms_ant = None, None, np.inf
    for it in range(int(MaxIt)):
        bu = x[1:] * (1 + eps) - x[0]
        bv = y[1:] * (1 + eps) - y[0]
        I0 = _pasoPOS(A, bu)
        J0 = _pasoPOS(A, bv)
        lam, mu = calculoLambdaMu(I0, J0)
        mejor = None
        for s in (1.0, -1.0):
            Is = I0 + s * lam * nu
            Js = J0 + s * mu * nu
            i, j, k, Z0, Rc, tc = calcularSalidaPOSIT(Is, Js, fx, (x[0], y[0]), M0)
            validos, rms = evaluarPose(Rc, tc, obj_vecs, img_vecs, K)
            if validos < n:
                continue
            if mejor is None or rms < mejor[0]:
                mejor = (rms, Rc, tc, k, Z0)
        if mejor is None:
            raise ValueError("aplicarPOSIT: ninguna solucion deja todos los puntos delante de la camara.")
        rms, R, t, k, Z0 = mejor
        eps = calcularAx(A, k) / Z0
        if abs(rms_ant - rms) < atol:
            break
        rms_ant = rms
    return R, t
