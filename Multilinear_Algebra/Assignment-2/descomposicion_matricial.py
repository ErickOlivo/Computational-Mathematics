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


def power_method(A, v, tol = 1e-3, max_iter = 10000):

    lambda0 = 0
    convyn = 0 # 0 no converge, 1 sí converge
    for i in range(max_iter):

        v = np.dot(A, v)
        v = v /np.linalg.norm(v)

        lambda1 = np.dot(np.transpose(v), np.dot(A, v))[0, 0]
        L = np.abs(lambda0 - lambda1)

        if L < tol:
            convyn = 1
            break
        lambda0 = lambda1

    if i == max_iter-1:
        print("No se alcanzó la convergencia con el número máximo de iteraciones")
    return convyn, lambda1, v


def schur_recursive(A, tol=1e-3):
    n = A.shape[0] #  Solo se usa filas, porque se trabaja con matrices cuadradas

    if n == 1:
        return np.eye(1), A.copy()

    # Paso 1: obtener valor propio dominante y vector propio
    v0 = np.random.rand(n, 1)
    _, lambda1, u = power_method(A, v0, tol=tol)

    # Paso 2: completar u a base ortonormal U = [u, V]
    u = u.reshape(-1, 1)
    V = np.eye(n)[:, 1:]
    U_aux = np.hstack([u, V])
    Q_U, _ = gram_schmidt_modificado(U_aux)
    U = Q_U

    # Paso 3: transformar A -> B = U^H A U (Cambio de base) dando lugar a una nueva matriz similar expresada en términos de los nuevos vectores base (U)
    B = U.conj().T @ A @ U # El primer valor propio aparecerá en la esquna superior izquierda B[0,0]

    # Paso 4: extraer submatriz (n-1)x(n-1) de abajo a la derecha
    B_sub = B[1:, 1:]

    # Paso 5: aplicar recursión a B_sub (n-1)x(n-1)
    # B se volverá más triangular en cada paso
    Q1, R1 = schur_recursive(B_sub)

    # Paso 6: construir Q_hat
    Q_hat = np.eye(n)
    Q_hat[1:,1:] = Q1

    # Paso 7: R = Q_hat^H B Q_hat
    R = Q_hat.conj().T @ B @ Q_hat

    # Paso 8: Q = Q_hat @ U
    Q = U @ Q_hat

    return Q, R

A = np.random.rand(4, 4)

Q, R = schur_recursive(A)

A_reconstruida = Q @ R @ Q.conj().T


print(f"A - Q R Q^H: {np.linalg.norm(A - A_reconstruida)}")

print(f"Q^H Q - I: {np.linalg.norm(Q.conj().T @ Q - np.eye(A.shape[0]))}")


