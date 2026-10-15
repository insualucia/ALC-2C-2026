import numpy as np

def norma(x, p):
  if p == 'inf':
      return float(np.max(np.abs(x)))
  return float(np.pow(np.sum(np.pow(np.abs(x), p)), (1/p)))

def normaliza(X, p):

  vectores_normalizados = []
  for v in X:
    norma_de_vector = norma(v,p)
    if norma_de_vector == 0:
      vectores_normalizados.append(v)
    else:
      vectores_normalizados.append(v/norma_de_vector)

  return vectores_normalizados



""" vectores_normalizados = []
  for x in X:
    n = norma(x, p)
    if n == 0:
      vectores_normalizados.append(x)
    else:
      vectores_normalizados.append(x/n)
  return vectores_normalizados"""

def norma_mat_mc(A, q, p, Np):

  max_norma = 0
  vector_norma_max = np.zeros(A.shape[1])
  for n in range(Np):
    x = np.random.randn(A.shape[1])
    norma_de_vector_x = norma(x, p)
    b = A @ x
    norma_de_vector_b = norma(b, q)
    norma_mat = norma_de_vector_b/norma_de_vector_x
    if max_norma < norma_mat:
      max_norma = norma_mat
      vector_norma_max = x

  return max_norma, vector_norma_max

def normaExacta(A, p: int | str | list = [1, 'inf']):
  if p == 1:
    return np.max(np.sum(np.abs(A), 0))
  elif p == 'inf':
    return np.max(np.sum(np.abs(A), 1))
  elif p == [1,'inf']:
    return [normaExacta(A, 1), normaExacta(A, 'inf')]
  else:
    return None

def condMC(A, p):                                                  # es (A, q, p) ?????????
  condicion = norma_mat_mc(A, p, p, 100000) * norma_mat_mc(np.linalg.inv(A), p, p, 100000)
  return condicion

def condExacto(A, p):
  return normaExacta(A, p) * normaExacta(np.linalg.inv(A), p)