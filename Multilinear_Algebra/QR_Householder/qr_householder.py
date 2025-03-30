import numpy as np

def householder(A):
    """
    Hace descomposición QR de una matriz A

    En luar de ortogonalizar columnas con proyecciones,
    refleja vectores para eliminar componentes debajo de la diagonal.

    Reflejar vectores, aplicando una transformación que borra las las partes debajo de la diagonal sin perder ortogonalidad

    Cuando se refleja un vector x respecto a un hiperplano,
    se genera un nuevo vector x' que es:
    Simétrico a x con respecto al hiperplano
    Está del otro lado, con la misma distancia al plano

    P = I - 2 w w^T
    Si w es un vector columna m x 1, entonces w^T es un vector fila 1 x m

    w w^T da una matriz m x m y se usa para reflejar cualquier vector x en ese espacio
    """



    X = A.copy()
    R = X.copy()

    n, m = X.shape

    # Comprobar que las columnas sean LI
    rank = np.linalg.matrix_rank(A)
    if rank != m:
        print("Matriz LD")
        return

    W = np.zeros((n, m))    # Almacena los vctores de relfexión w_k

    Q = np.zeros((n, m))

    for k in range(0, m):
        r_k = R[:, k].copy()

        # if k > 0:
            # r_k = R[:, k].copy()

        # Aplicar las reflexiones anteriores P_1, P_2, ..., P_{k-1} a la columna x_k para obtener r_k
        for i in range(k):
            w_i = W[:, i] #  w_k es el vector de Householder calculado en el paso k

            """
            r_k := P_{k-1}...P_{1}x_k eso es equivalente a P_{i} x = x - 2 w_i (w_i^T x)

            Interesa el efecto de P_i sobre un vector, no realmente P_i
            Si se usara np.eye(n) sería ineficiente, se crearía una matriz grande

            Así se aplica la reflexión P_i sin tener que construir P_i omo matriz

            """
            r_k = r_k - 2 * w_i * (w_i.T @ r_k) # P = I - 2 w w^T


        # Paso 4: Compute w_k

        # Extraer subvector
        x = r_k[k:] # El vector desde el elemento k hasta el final.

        # Calcular Beta (1.21)
        """
        X[0] es r_k[k] es decir elemento 0 en la posición k
        x[1] sería r_k[k+1]

        np.linalg.norm(x) es ||r_k[k:]||_2
        """
        """
        El valor beta se utiliza para construir el vector w_k
        Este se ajusta según el signo del primer componente de x

        """
        beta = np.sign(x[0]) * np.linalg.norm(x)

        # Calcular vector z (1.20)
        """
        z = [ 0, 0, 0, r_k[k] + beta, r_k[k+1], r_k[k+2], ... ]
            ↑      ↑
            i<k     i = k     (y demás i > k)

        """
        z = r_k.copy()          # si i > k
        z[:k] = 0               # Si i < k, poner zeros en todas las posiciones anteriores a k
        z[k] = r_k[k] + beta    # si i = k, modificamos el valor de la posición k en z sumando beta


        # Normalizar z para obtener w_k, es el que se usa para la reflexión en la matriz R
        w_k = z / np.linalg.norm(z)
        # Guardar w_k en la matriz W
        W[:, k] = w_k


        # Paso 5: Reflexión a toda la matriz R, no solo a una columna por eso no se usa R[:, k]

        # R := R - 2 w_k (w_k.T R)
        """
        Si w_k es un vector columna (nx1)
        Y v = w_k.T @ R es una ila (1xm)
        np.outer(w_k, v) da una matriz (nxm)
        """
        R = R - 2 * np.outer(w_k, w_k.T @ R) # Utilizamos el producto externo para calcular la actualización de R en función del vector w_k

        # Paso 6: q_k = P_1 P_2 ... P_k e_k
        q_k = np.zeros(n)       # Inicializar vector de ceros del tamaño de filas de la matriz A
        q_k[k] = 1              # e_k, eso corresponde al vector estándar

        # Aplicar reflexiones en orden inverso: P_k, P_{k-1}, ..., P_0
        """
        Esto es necesario porque las reflexiones en P_k afectan no solo a la columna k de la matriz,
        sino también a las columnas posteriores.
        Para obtener Q, aplicamos transformaciones en el orden inverso al que calculamos.
        """
        for i in reversed(range(k+1)):
            w_i = W[:, i]
            q_k = q_k - 2 * w_i * (w_i.T @ q_k)

        # Guardar q_k como columna k de Q
        Q[:, k] = q_k

    R = R[:m, :] # m primeras filas, se recortan las primeras m filas, R debe tener el mismo número de filas y columnas que la matriz A
    return Q, R

A = np.random.rand(3, 2)

Q, R = householder(A)

print(f"A - QR = ", np.linalg.norm(A - Q@R))

print(f"Q^T Q = ", np.allclose(Q.T @ Q, np.eye(Q.shape[1])))


