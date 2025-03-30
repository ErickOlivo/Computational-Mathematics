import numpy as np

def gram_schmidt(A):
    m, n = A.shape

    """
    # No hace falta esto, ya que se detecta LD en el bucle

    rank = np.linalg.matrix_rank(A)
    if rank != n:
        print("Matriz LD")
        return
    """

    R = np.zeros((m,n)) #nxn para que sea triangular superior y cuadrada

    Q = np.zeros((m,m)) #mxn para poder almacenar las columnas correctamente


    R[0,0] = np.linalg.norm(A[:, 0])
    if R[0,0] == 0:
        print("Primer vector columna es LD")
        return
    else:
        Q[:, 0] = A[:, 0] / R[0,0]

    for j in range(1, n):
        sumatoria_projeccion = np.zeros(m)
        for i in range(j):
            R[i,j] = np.dot(A[:, j], Q[:, i])
            sumatoria_projeccion += R[i,j] * Q[:, i]

            # Ahora construyo un nuevo vector restando a x_j las proyecciones sobre los vectores q_i anteriores
            # Q_hat es un vector ortogonal a todos los q_i previos, pero aún no está normalizado
            # Se quita de x_j todo lo que ya está en dirección de los vectores anteriores, para solo quedarme con la parte perpendicular a ellos
            # Donde (x_j, q_i) es la proyección de x_j sobre q_i

        q_hat = A[:, j] - (sumatoria_projeccion)

        """
        Paso 5
        """
        R[j,j] = np.linalg.norm(q_hat)

        """
        Paso 6
        """
        if R[j,j] == 0:
            print("Stop")
            print("Vector LD")
            return
        else:
            Q[:, j] = q_hat / R[j,j]

    return Q, R



A = np.random.rand(4, 3)
Q, R = gram_schmidt(A)


print(f"A - QR = ", np.linalg.norm(A - Q@R))










