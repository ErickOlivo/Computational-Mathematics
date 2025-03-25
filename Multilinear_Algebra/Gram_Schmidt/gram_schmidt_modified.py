import numpy as np

def gram_schmidt_modificado(X):
    """
    Entrada:
    X (np.array): Matriz de entrada donde cada columna es un vector.

    Salida:
    Q (np.array): Matriz con columnas ortonormales.
    R (np.array): Matriz triangular superior.
    """
    m, n = X.shape
    Q = X.copy().astype(float)  # Copiar la matriz original
    R = np.zeros((n, n))

    for j in range(n):
        # Paso 1: Calcular la norma del j-ésimo vector
        R[j, j] = np.linalg.norm(Q[:, j])
        if np.isclose(R[j, j], 0):
            raise ValueError(f"Vectores linealmente dependientes en columna {j+1}.")

        # Paso 2: Normalizar el j-ésimo vector para obtener q_j
        Q[:, j] = Q[:, j] / R[j, j]

        # Paso 3: Ortogonalizar los vectores posteriores contra q_j
        for i in range(j+1, n):
            R[j, i] = np.dot(Q[:, j], Q[:, i])  # Producto interno
            Q[:, i] = Q[:, i] - R[j, i] * Q[:, j]  # Actualización inmediata

    return Q, R



X = np.array([[1, 1],
              [1, 0]], dtype=float).T  # Vectores en columnas

Q, R = gram_schmidt_modificado(X)

print("Q (ortonormal):\n", np.round(Q, 4))
print("\nR (triangular superior):\n", np.round(R, 4))
print("\nVerificación QR ≈ X:\n", np.round(Q @ R, 2))
