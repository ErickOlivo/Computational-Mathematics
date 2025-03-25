import numpy as np

def householder_qr(A):
    """
    Entrada:
    A (np.array): Matriz de entrada de tamaño m x n.

    Salida:
    Q (np.array): Matriz ortogonal (m x m).
    R (np.array): Matriz triangular superior (m x n).
    """
    m, n = A.shape
    R = A.copy().astype(float)  # Copia de A para no modificar la original
    Q = np.eye(m)              # Matriz identidad para acumular Q

    for k in range(min(m, n)):
        # Paso 1: Extraer la columna k desde la diagonal hacia abajo
        x = R[k:, k].copy()

        # Paso 2: Calcular la norma de x con signo opuesto al de x[0]
        sigma = -np.sign(x[0]) * np.linalg.norm(x)

        # Paso 3: Calcular el vector v = x - sigma*e1
        e1 = np.zeros_like(x)
        e1[0] = 1
        v = x - sigma * e1

        # Paso 4: Normalizar v (beta = 2/v^T v)
        if np.linalg.norm(v) < 1e-12:
            continue  # Evitar división por cero
        beta = 2.0 / (v @ v)

        # Paso 5: Aplicar el reflector a R
        R[k:, k:] = R[k:, k:] - beta * np.outer(v, v @ R[k:, k:])

        # Paso 6: Acumular el reflector en Q
        Q[:, k:] = Q[:, k:] - beta * (Q[:, k:] @ v) @ v.T

    # Triangularizar R (eliminar ceros debajo de la diagonal)
    R = np.triu(R[:n, :]) if m > n else np.triu(R)

    return Q, R


A = np.array([[1, 1], [1, 0]], dtype=float)
Q, R = householder_qr(A)

print("Q:\n", np.round(Q, 4))
print("\nR:\n", np.round(R, 4))
print("\nVerificación QR ≈ A:\n", np.round(Q @ R, 2))
