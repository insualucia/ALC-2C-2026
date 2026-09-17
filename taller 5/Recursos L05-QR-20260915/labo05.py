import numpy as np

def proyector_ortogonal(u,v): # Devuelve P_u(v)

    if len(u) != len(v):
        return None

    producto_interno_v_u = 0
    for i in range(len(u)):
        producto_interno_v_u = u[i]*v[i] + producto_interno_v_u

    norma_2_u = norma(u,2)

    return (producto_interno_v_u/norma_2_u) @ u

def producto_interno(u,v):

    if len(u) != len(v):
        return None

    acc = 0
    for i in range(len(u)):
        acc = acc + u[i] * v[i]

    return acc
    

def QR_con_GS(A, tol = 1e-12, retorna_nops = False):

    Q, R = np.zeros((len(A), len(A)))

    if not esCuadrada(A):
        return None

    norma_2_q_1 = norma(A[:0],2)
    Q[:0] = Q[:0] @ norma_2_q_1
    for j in range(1, len(A)):
        qj_monio = A[:j]
        for k in range(j-1):
            R[k][j] = producto_interno(Q[:k],qj_monio)
            qj_monio = qj_monio - Q[:k] @ R[k][j]
        R[j][j] = norma(qj_monio,2)
        Q[:j] = qj_monio @ (1/R[j][j])



     

if __name__ == "__main__":

    0