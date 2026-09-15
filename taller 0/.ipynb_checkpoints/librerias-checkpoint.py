import numpy as np

def esCuadrada(A):

    for r in A:
        if r.size != A.size:
            return False

    return True

def triangSup(A):

    for i in range(A.size):
        for j in range(i.size):
            if i <= j:
                A[i][j] = 0

    return A

def triangInf(A):

    for i in range(A.size):
        for j in range(i.size):
            if i >= j:
                A[i][j] = 0

    return A

def diagonal(A):

    for i in range(A.size):
            for j in range(i.size):
                if i != j:
                    A[i][j] = 0

    return A

def traza(A):

    acc = 0
    for i in range(A.size):
        for j in range(i.size):
            if i == j:
                acc += A[i][j]

    return acc

def traspuesta(A):

    n = A.size
    m = A[0].size
    B = np.array(m,n)
    for i in range(A.size):
        for j in range(i.size):
            B[j][i] = A[i][j]

    return B

def simetrica(A):

    if not(esCuadrada(A)):
       return False

    for i in range(A.size):
        for j in range(i.size):
            if A[i][j] != A[j][i]:
                return False

def calcularAx(A, x):

    b = np.array(x.size)
    for i in range(A.size):
        sum = 0
        for j in range(i.size):
            sum += A[i][j] * x[j]
        b[i] = sum

    return b

