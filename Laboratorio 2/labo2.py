import numpy as np

def rota(angulo):
  T = np.array([
    [np.cos(angulo), -1 * np.sin(angulo)],
    [np.sin(angulo), np.cos(angulo)]])
  return T

def escala(s):
    A = np.zeros([len(s), len(s)])
    for i in range(len(s)):
      A[i][i] = s[i]
    return A

def rota_y_escala(angulo, s):
  E = escala(s)
  R = rota(angulo)
  assert(R.shape[1] == E.shape[0])
  A = E @ R
  return A

def afin(theta,s,b):
  RyE = rota_y_escala(theta, s)
  A = np.zeros([3,3])
  for i in range(len(RyE)):
    for j in range(len(RyE[i])):
      A[i][j] = RyE[i][j]
  A[0][2] = b[0]
  A[1][2] = b[1]
  A[2][2] = 1
  return A

def calcular_ax(A,x):
  n = x.shape[0]
  res = np.zeros(n)

  for i in range(n):
    for j in range(n):
      res[i] +=  A[i][j] * x[j]
  return res

def trans_afin(v, theta, s, b):
  F = afin(theta, s, b)
  w = calcular_ax(F, v)
  w[0] += F[0][2]
  w[1] += F[1][2]
  return w