import numpy as np

def QR_con_GS(A, tol=1e-12,retorna_nops=False):
  n = A.shape[0] 
  m = A.shape[1]
  if n != m:
    return None

  Q = np.zeros((n, m))
  R = np.zeros((n, m))

  for j in range(m):
    u = A[:,j]
    for i in range(j):
      R[i,j] = np.dot(Q[:,i], A[:,j])
      u = u - R[i,j] * Q[:,i]

    normaU = norma(u, 2)

    if normaU < tol:
      R[j,j] = 0.0
      Q[:,j] = np.zeros(n)
    else:
      R[j,j] = normaU
      Q[:,j] = u / normaU

  return Q, R

#def productoInterno(v1, v2):
 # suma = 0
  #n = v1.shape[0]
  #for i in range(n):
  #  suma += v1[i] * v2[i]
  #return suma

def norma(x, p):
  if p == 'inf':
    return np.max(np.abs(x))
  return np.pow(np.sum(np.pow(np.abs(x), p)), (1/p))

def QR_con_HH(A, tol=1e-12, extras=False):
  m = A.shape[0] 
  n = A.shape[1]
  if n > m:
    return None

  Q = np.eye(m)
  R = A.copy()

  for j in range(n):
    columnaDesdeDiag = R[j:, j]
    norma_col = norma(columnaDesdeDiag, 2)

    if norma_col >= tol:
      if columnaDesdeDiag[0] != 0:
        alfa = -np.sign(columnaDesdeDiag[0]) * norma_col  
      else:
        alfa = -np.sign(columnaDesdeDiag[0]) * -norma_col 

    v = columnaDesdeDiag.copy()
    v[0] -= alfa
    norma_v = norma(v, 2)

    if norma_v > tol:
      v = v / norma_v

      R[j:, j:] -= 2.0 * multiplicacionDeVectoresTraspuestos(v, np.dot(v, R[j:, j:]))
      Q[:, j:] -= 2.0 * multiplicacionDeVectoresTraspuestos(np.dot(Q[: ,j:], v), v) 

  for i in range(m):
    for j in range(min(i ,n)):
      if abs(R[i, j]) < tol:
        R[i, j] = 0.0

  return Q, R

def multiplicacionDeVectoresTraspuestos(v1, v2):
  n = v1.size
  m = v2.size
  A = np.zeros((n, m))
  for i in range(n):
    for j in range(m):
      A[i,j] = v1[i] * v2[j] 
  return A

def calculaQR(A,metodo='RH', tol=1e-12):
  if metodo == 'RH':
      return QR_con_HH(A, tol=tol)
  elif metodo == 'GS':
      return QR_con_GS(A, tol=tol)
  else:
      return None
      
# Tests L05-QR:

# --- Matrices de prueba ---
A2 = np.array([[1., 2.],
               [3., 4.]])

A3 = np.array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 0.]])

A4 = np.array([[2., 0., 1., 3.],
               [0., 1., 4., 1.],
               [1., 0., 2., 0.],
               [3., 1., 0., 2.]])

# --- Funciones auxiliares para los tests ---
def check_QR(Q,R,A,tol=1e-10):
    # Comprueba ortogonalidad y reconstrucción
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
    assert np.allclose(Q @ R, A, atol=tol)

# --- TESTS PARA QR_by_GS2 ---
Q2,R2 = QR_con_GS(A2)
check_QR(Q2,R2,A2)

Q3,R3 = QR_con_GS(A3)
check_QR(Q3,R3,A3)

Q4,R4 = QR_con_GS(A4)
check_QR(Q4,R4,A4)

# --- TESTS PARA QR_by_HH ---
Q2h,R2h = QR_con_GS(A2)
check_QR(Q2h,R2h,A2)

Q3h,R3h = QR_con_HH(A3)
check_QR(Q3h,R3h,A3)

Q4h,R4h = QR_con_HH(A4)
check_QR(Q4h,R4h,A4)

# --- TESTS PARA calculaQR ---
Q2c,R2c = calculaQR(A2,metodo='RH')
check_QR(Q2c,R2c,A2)

Q3c,R3c = calculaQR(A3,metodo='GS')
check_QR(Q3c,R3c,A3)

Q4c,R4c = calculaQR(A4,metodo='RH')
check_QR(Q4c,R4c,A4)