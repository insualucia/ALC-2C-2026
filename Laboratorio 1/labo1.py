import numpy as np

def error(x, y):
  y = np.float64(y)
  return abs(x - y)

def error_relativo(x, y):
  y = np.float64(y)
  return abs((x - y)/x)

def son_iguales(x,y,atol=1e-08):
  return np.allclose(error(x,y),0,atol=atol)