import numpy as np

def gram_schmidt_estandar(X):
    """
    Entrada:
    X (np.array): Matriz de entrada donde cada columna es un vector.

    Salida:
    Q (np.array): Matriz con columnas ortonormales.
    R (np.array): Matriz triangular superior.
    """
    m, n = X.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    # Paso 1: Procesar el primer vector
    R[0, 0] = np.linalg.norm(X[:, 0])
    if np.isclose(R[0, 0], 0):
        raise ValueError("El primer vector es nulo.")
    Q[:, 0] = X[:, 0] / R[0, 0]

    # Paso 2: Procesar vectores restantes
    for j in range(1, n):
        v = X[:, j].copy()  # Copiar el j-ésimo vector

        # Calcular proyecciones sobre los q_i anteriores
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], X[:, j])
            v -= R[i, j] * Q[:, i]

        # Normalizar el vector resultante
        R[j, j] = np.linalg.norm(v)
        if np.isclose(R[j, j], 0):
            raise ValueError(f"Vectores linealmente dependientes en columna {j+1}.")
        Q[:, j] = v / R[j, j]

    return Q, R


X = np.array([[1, 1],
              [1, 0]], dtype=float).T

Q, R = gram_schmidt_estandar(X)

print("Q (ortonormal):\n", np.round(Q, 4))
print("\nR (triangular superior):\n", np.round(R, 4))
print("\nVerificación QR ≈ X:\n", np.round(Q @ R, 2))
