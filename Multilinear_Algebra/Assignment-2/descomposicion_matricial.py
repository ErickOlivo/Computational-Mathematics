import numpy as np

def gram_schmidt_modificado(X):
    """
    Entrada:
    X (np.array): Matriz de entrada donde cada columna es un vector.

    Salida:
    Q (np.array): Matriz con columnas ortonormales.
    R (np.array): Matriz triangular superior.
    """
    if X.size == 0:
        raise ValueError("La matriz de entrada está vacía")

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

def cociente_rayleigh(A, x):
    """
    Entrada:
        A: Matriz cuadrada (numpy array).
        x: Vector no nulo (numpy array).

    Salida:
        Valor del cociente (float).
    """
    return np.dot(x.T, np.dot(A, x)) / np.dot(x.T, x)


def metodo_potencia(A, v=None, tol = 1e-6, max_iter = 5000):
    """
    Entrada:
        A: Matriz cuadrada.
        tol: Tolerancia para convergencia (opcional).
        max_iter: Máximo de iteraciones (opcional).
        v: Vector inicial, si no se especifica se genera aleatorio

    Salida:
        converged: Bool indicando si convergió.
        lambda_est: Valor propio dominante estimado.
        x: Vector propio asociado (normalizado).
    """
    n = A.shape[0]
    if v is None:
        v = np.ones(n)
    v = v / np.linalg.norm(v)  # Normalizar el vector inicial
    lambda_prev = 0
    converged = False

    for i in range(max_iter):
        v_new = A @ v
        v_new = v_new / np.linalg.norm(v_new)
        lambda_new = cociente_rayleigh(A, v_new)  # Usando tu función

        if np.abs(lambda_new - lambda_prev) < tol:
            converged = True
            break

        lambda_prev, v = lambda_new, v_new

    if not converged:
        print("No se alcanzó la convergencia con el número máximo de iteraciones.")
    return converged, lambda_new, v_new



def descomposicion_schur(A, tol=1e-10, max_iter=10000):
    """
    Entrada:
        A: Matriz cuadrada (numpy array).
        tol: Tolerancia para valores propios.

    Salida:
        Q: Matriz unitaria.
        T: Matriz triangular superior.
    """

    n = A.shape[0]
    if n == 1:
        return np.array([[1.0]]), A.copy()

    # Paso 1: Encontrar vector propio dominante
    _, lambda_val, v = metodo_potencia(A, tol=tol, max_iter=max_iter)
    
    '''
    v.reshape(-1, 1) transforma un vector fila o columna en una matriz de una sola columna
    En este caso se convierte en un vector columna (n,1)
    '''
    v = v.reshape(-1, 1) / np.linalg.norm(v) # Vector propio dominante normalizado

    # Paso 2: Completar base ortonormal de forma más estable
    Q = np.eye(n)
    
    Q[:, 0] = v.flatten() # Primera columna = vector propio
    for i in range(1, n):
        '''
        Se normaliza la primera columna Q[:,0]
        Para cada columna se resta la proyección de Q[:,j] sobre todas las columnas anteriores Q[:,0],...,[Q[:,j-1]
        '''
        Q[:, i] = np.random.randn(n) # Rellena con vector aleatorios
    Q, _ = gram_schmidt_modificado(Q) # Ortogonaliza toda la matriz

    # Paso 3: Calcular T = Q^T A Q
    T = Q.T @ A @ Q

    # Paso 4: Llamada recursiva
    Q_sub, T_sub = descomposicion_schur(T[1:, 1:], tol, max_iter)

    # Paso 5: Ensamblar matrices finales
    Q_final = Q @ np.block([[1, np.zeros((1, n-1))],
                           [np.zeros((n-1, 1)), Q_sub]])
    T_final = np.block([
        [T[0, 0], T[0, 1:].reshape(1, -1) @ Q_sub],
        [np.zeros((n-1, 1)), T_sub]])

    return Q_final, T_final





A = np.array([[4, 2, 1],
              [0, 3, -1],
              [1, 0, 2]])

# 1. Factorización QR
Q, R = gram_schmidt_modificado(A)
print("Factorización QR:")
print("Q:\n", Q.round(4))
print("R:\n", R.round(4))
print("¿Q ortogonal?:", np.allclose(Q.T @ Q, np.eye(3)), "\n")

# 2. Método de la potencia
converged, lambda_dom, x_dom = metodo_potencia(A, tol=1e-8, max_iter=5000)
print("Método de la potencia:")
print("Convergencia:", converged)
print("Valor propio dominante:", lambda_dom.round(6))
print("Vector propio asociado:\n", x_dom.round(6), "\n")

# 3. Descomposición de Schur
Q_schur, T_schur = descomposicion_schur(A)
print("Descomposición de Schur:")
print("Q (unitaria):\n", Q_schur.round(4))
print("¿Q unitaria?:", np.allclose(Q_schur @ Q_schur.T, np.eye(3)))
print("T (triangular superior):\n", T_schur.round(4))
print("¿A = Q T Q^T?:", np.allclose(A, Q_schur @ T_schur @ Q_schur.T))
