import numpy as np

def gram_schmidt_modificado(A):
    m, n = A.shape

    Q = np.zeros((m,n))
    R = np.zeros((n,n))


    R[0,0] = np.linalg.norm(A[:, 0])

    if R[0,0] < 1e-12:
        print("Stop")
        print("Primer vector es LD")
        return

    else:
        Q[: ,0] = A[:, 0] / R[0,0] # Asigno la columna completa del vector normalizado como q_{1}


    for j in range(1, n):
        q_hat = A[:, j]

        """
        Ya no hay una sumatoria explícita, en vez de acumular y luego restar:
        Proyección sobre Q[:, i]
        Restar enseguida esa proyección
        Seguir al siguiente Q[:, i]

        Esa actualización inmediata en cada paso ayuda a mantener la ortogonalidad
        """
        for i in range(j):
            R[i,j] = np.dot(q_hat, Q[:, i]) # Q[:, i] porque se busca el i-ésimo vector columna
            q_hat = q_hat - R[i,j]*Q[:, i]

        R[j,j] = np.linalg.norm(q_hat)

        if R[j,j] < 1e-12: # No pongo if R[j,j] == 0, para evitar errores de redondeo
            print("Stop")
            print("Vector columna LD")
            return
        else:
            Q[:, j] = q_hat / R[j,j]

    return Q, R

A = np.random.rand(4, 3)

Q, R = gram_schmidt_modificado(A)


print(f" A - QR = ", np.linalg.norm(A - Q@R))
