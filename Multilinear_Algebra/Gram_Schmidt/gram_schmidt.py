import numpy as np

def gram_schmidt(X):
    """
    Implementa el algoritmo de Gram–Schmidt clásico de forma muy cercana
    al pseudocódigo original.

    Parámetros
    ----------
    X : list of np.ndarray
        Lista (o array) de vectores. Cada elemento X[j] es un vector de R^n.
        Por ejemplo, X podría ser [x1, x2, ..., xr].

    Retorna
    -------
    Q : list of np.ndarray
        Lista de vectores ortonormales resultantes [q1, q2, ..., qr].
    R : np.ndarray
        Matriz (r x r) con los coeficientes r_ij. Es triangular superior.
    """

    # Número de vectores
    r = len(X)

    # Matriz R para guardar los coeficientes r_ij
    # Al final, R será de dimensiones r x r (o las que se necesiten).
    R = np.zeros((r, r))

    # Lista para los vectores ortonormales q1, q2, ...
    Q = []

    # ------------------------- PASO 1 --------------------------
    # "Compute r_11 = ||x1||_2. If r_11 = 0 Stop, else compute q1 = x1 / r_11"
    x1 = X[0]             # primer vector
    r_11 = np.linalg.norm(x1, 2)  # norma 2 de x1
    R[0, 0] = r_11

    # Umbral para considerar "cero" por cuestiones numéricas
    tol_cero = 1e-14

    if abs(r_11) < tol_cero:
        # Si no hay norma, no hay nueva dirección, paramos
        return [], R
    else:
        # Normalizamos x1
        q1 = x1 / r_11
        Q.append(q1)      # guardamos q1 en nuestra lista

    # ------------------------- PASO 2 --------------------------
    # "For j = 2, ..., r Do:"
    # (en Python, índice j=1 corresponde al "segundo" vector,
    #  así que iremos de j=1 a r-1 en 0-based index)
    for j in range(1, r):
        # Tomamos el vector x_j
        x_j = X[j].copy()  # conviene copiar para no mutar el original

        # ---------- PASO 3 ----------
        # "Compute r_ij := ( x_j , q_i ) for i = 1, 2, ..., j-1"
        # Nótese que en pseudocódigo i va de 1 a j-1,
        # pero en Python arrays i va de 0 a j-1
        for i in range(j):
            r_ij = np.dot(x_j, Q[i])  # producto interno ( x_j , q_i )
            R[i, j] = r_ij

            # "q_hat := x_j - sum_{i=1}^{j-1} r_ij * q_i"
            # De manera incremental, vamos restando cada proyección
            x_j = x_j - r_ij * Q[i]

        # ---------- PASO 4 y 5 ----------
        # "r_jj := ||q_hat||_2" y "if r_jj = 0 then Stop, else q_j = q_hat / r_jj"
        r_jj = np.linalg.norm(x_j, 2)
        R[j, j] = r_jj

        if abs(r_jj) < tol_cero:
            # No hay dirección independiente, paramos
            break
        else:
            q_j = x_j / r_jj
            Q.append(q_j)

    return Q, R

