import numpy as np

def norma(x, p):
  if p == 'inf':
      return np.max(np.abs(x))
  return np.pow(np.sum(np.pow(np.abs(x), p)), (1/p))

def normaliza(X, p):
  vectores_normalizados = []
  for x in X:
    n = norma(x, p)
    if n == 0:
      vectores_normalizados.append(x)
    else:
      vectores_normalizados.append(x/n)
  return vectores_normalizados

def normaMatMC(A, q, p, Np):
  n = A.shape[1]
  vectores_aleatorios = [np.random.randn(n) for i in range(Np)]
  vectores_normalizados = normaliza(vectores_aleatorios, p)
  normas_transformadas = [norma(A@x, q) for x in vectores_normalizados]
    
  indice_max = np.argmax(normas_transformadas)  
  norma_max = normas_transformadas[indice_max]
  x_max = vectores_normalizados[indice_max]
  return norma_max, x_max

def normaExacta(A, p=[1, 'inf']):
  if p == 1:
    return np.max(np.sum(np.abs(A), axis=0))
  elif p == 'inf':  
    return np.max(np.sum(np.abs(A), axis=1))
  else:
    return None

#def condMC(A, p):                                                  # es (A, q, p) ?????????
 # condicion = normaMatMC(A, 1, p, 100000) * normaMatMC(np.linalg.inv(A), 1, p, 100000)
  #return condicion

assert(np.allclose(norma(np.array([1,1]),2),np.sqrt(2)))
assert(np.allclose(norma(np.array([1]*10),2),np.sqrt(10)))
assert(norma(np.random.rand(10),2)<=np.sqrt(10))
assert(norma(np.random.rand(10),2)>=0)

for x in normaliza([np.array([1]*k) for k in range(1,11)],2):
    assert(np.allclose(norma(x,2),1))
for x in normaliza([np.array([1]*k) for k in range(2,11)],1):
    assert(not np.allclose(norma(x,2),1) )
for x in normaliza([np.random.rand(k) for k in range(1,11)],'inf'):
    assert( np.allclose(norma(x,'inf'),1) )

assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]),1),2))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),1),6))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),'inf'),7))
assert(normaExacta(np.array([[1,-2],[-3,-4]]),2) is None)
assert(normaExacta(np.random.random((10,10)),1)<=10)
assert(normaExacta(np.random.random((4,4)),'inf')<=4)

nMC = normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
assert(np.allclose(nMC[0],1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))

nMC = normaMatMC(A=np.eye(2),q=2,p='inf',Np=100000)
assert(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))

A = np.array([[1,2],[3,4]])
nMC = normaMatMC(A=A,q='inf',p='inf',Np=100000) # cambié Np de 1 millon a 100 mil para que corra mas rapido
assert(np.allclose(nMC[0],normaExacta(A,'inf'),rtol=2e-1))

'''A = np.array([[1,1],[0,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA_ = normaMatMC(A_,2,2,10000)
condA = condMC(A,2,10000)                                           # es (A, q, p) ?????????
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-3))

A = np.array([[3,2],[4,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA_ = normaMatMC(A_,2,2,10000) 
condA = condMC(A,2,10000)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-3))'''