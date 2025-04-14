import numpy as np
def qr_iterative(A, tol=1e-8, max_iter=1000):
    """
    Permita transformar una matriz en una forma de Schur en la cual sus autovalores aparecen en la diagonal.
    A medida que el proceso converge, la matriz resultante muestra estos valores en forma clara.

    Cuando se decompone A = QR y luego se recompone A_k+1 = RQ, se puede demostrar que A_k+1 = Q^T A Q, como Q es ortogonal
    Q^T Q = I. Esto significa que A_k+1 es similar a A, por lo tanto tienen los mismos autovalores.

    Este proceso iterativo "empuja" los valores propios hacia la diagonal

    Por el teorea de Schur si A es real garantiza que A es ortogonalmente similar a una matriz que es casi triangular
    Q^T A Q = T
    """
    n = A.shape[0]
    U = np.eye(n)    # Acumulador para las transformaciones

    iter = 0

    """
    Continúa hasta que A_k converja a una forma triangular (real) o quasi_triangular por bloques (si hay valores complejos)

    Los valores de la diagonal de la matriz convergida (A_k final) son las aproximaciones a los autovalores
    """
    while iter < max_iter:
        # Paso 1: Realizar la descomposición QR de la matriz actual
        Q, R = np.linalg.qr(A)
        # Acumular la transformación U (para obtener la matriz ortogonal necesaria)
        U = np.dot(U, Q)

        # Paso 2: Actualizar la matriz
        A = np.dot(R, Q)

        # Paso 3: Evaluar el criterio de parada
        # Extraer la parte diagonal de A
        # np.diag(A) extrae los elementos de la diagonal y devuelve un arreglo 1D con dichos elementos.
        # np.diag(...) se construye de nuevo una matriz diagonal usando esos elementos.
        D = np.diag(np.diag(A))
        # Calcular la norma L1 de los elementos fuera de la diagonal
        error = np.sum(np.abs(A - D)) # Determina cuándo la parte no diagonal es suficientemente pequeña
        # La idea es que cuando esta suma es muy pequeña podemos decir que los elementos fuera de la diagonal han sido reducidos a casi cero.
        if error < tol:
            break

        iter += 1

    return U, A # Al salir, A estará casi diagonal y U contiene la acumulación de los Q's, esta U es ortogonal (o unitario en el caso complejo) y contiene los vectores singulares izquierdos de A
                # Una matriz A que es casi triangular (o diagonal si el proceso converge completamente). Los elementos de la diagonal son aproximaciones a los autovalores originales
                # En un problema estándar de autovalores, la Schur normal es A = U T U^T si es real


# ---------------
# Código principal
# ---------------

# Paso A: Definir la matriz A original
A = np.random.rand(4, 4)

# Hacer una copia para el proceso QR (ya que se modifica A en el proceso)
A_copy = A.copy()

# Paso B: Aplicar QR iterativo para obtener U y la matriz casi diagonal T (denominada A_diag)
U, A_diag = qr_iterative(A_copy) # A_diag is T

# Imprimir resultados parciales
print("Matriz U (acumulada de Q's):")
print(U)
print("\nMatriz casi diagonal (T):")
print(A_diag)

# Comprobación Schur: Reconstruir usando U y T
A_reconstructed = U @ A_diag @U.conj().T

print(f"Error de reconstrucción de Schur: A - U T U^H = {np.linalg.norm(A - A_reconstructed)}")

# --------
# Paso C: Obtener V y los valores singulares (mu_i) usando QR iterativo aplicado a A^T A
# --------

"""
Obtener una matriz V y los valores u_i (que serán los valores singulares) tal que U^T A V = diag(u_i)

La función qr_iterative tranforma A en una forma casi triangular T en la que los autovalores (o, en el contexto de la SVD, los cuadrados de los valores singulares) aparecen en la diagonal
Los valores singulares se obtienen como u_i = sqrt(lamda_i) con lambda_i autovalores de A^T A
"""


# Definimos B = A^T A
B = A.T @ A

# Aplicamos la misma función QR iterativo a B para obtener V (acumulador de Q's para B) y la forma casi diagonal B_diag
V, B_diag = qr_iterative(B)

# los autovalores de B se aproximan a la diagonal de B_diag
# Se extraen de B_diag, y no de A_diag ya que que los autovalores de A no se relacionan de manera directa con los valores signulares
lambda_vals = np.diag(B_diag)

# Calcular los valores singulares de A: mu_i = sqrt(lambda_i) (asegurándonos de que sean no negativos)
# np.maximun(lambda_vals, 0) compara cada valor de lambda_i con 0 y devuelve el mayor de ambos, de esa manera si por errores numéricos y de redondeo lambda_i es ligeramente negativo, se usará 0 en su lugar.
# Así se asegura estar calculando la raíz cuadrada de números no negativos
mu = np.sqrt(np.maximum(lambda_vals, 0))

print("\nValores singulares (mu):")
print(mu)
print("\nMatriz V (obtenida a partir del QR iterativo aplicado a A^T A):")
print(V)


# --------------------------
# Paso D: Validar la decomposición completa
# Se desea que U^T A V sea diagonal con los valores mu en la diagonal.
# --------------------------

# Calcular M inicial
M = U.T @ A @ V



# Ajustar los signos de las columnas de U en función de la diagonal de M:
diag_M = np.diag(M)
for i in range(len(diag_M)):
    if diag_M[i] < 0:
        # Cambiamos el signo de la columna i de U para invertir el signo correspondiente en M.
        U[:, i] = -U[:, i]


# Recalcular M con los signos ajustados:
M = U.T @ A @ V

# Ahora, para la validación:
# Se compara la matriz M con la matriz diagonal esperada,
# que es np.diag(μ). Como la diagonal de M puede haber quedado con pequeños errores numéricos,
# medimos:
#  - La diferencia entre los valores absolutos de la diagonal de M y μ (error_diag)
#  - La suma de las entradas fuera de la diagonal (error_offdiag)
M_diag = np.diag(np.diag(M))
mu_diag = np.abs(np.diag(M))  # Consideramos los valores absolutos para comparar con μ

error_final = np.sum(np.abs(M_diag - np.diag(mu)))


print("\nM = U^T A V:")
print(M)
print("\nMatriz diagonal extraída de M (con los valores ajustados):")
print(M_diag)
print(f"\nError final (||M - diag(μ)||_1): {error_final}")

epsilon = 1e-20
if error_final < epsilon:
    print("\nLa descomposición cumple: ||U^T A V - diag(μ)||_1 < tol")
else:
    print("\nLa descomposición No cumple la condición deseada.")








"""
# Extraer la parte diagonal de M
M_diag = np.diag(np.diag(M))

# Calcular el error L1 de la diferencia
error_final = np.sum(np.abs(M - M_diag))




mu_diag = np.abs(np.diag(M))
error_diag = np.sum(np.abs(mu_diag - mu))
error_offdiag = np.sum(np.abs(M - np.diag(np.diag(M))))
error_final = error_diag + error_offdiag



print("\nM = U^T A V:")
print(M)
print("\nMatriz diagonal extraída de M:")
print(M_diag)
print(f"\nError final (||M - diag(M)||_1): {error_final}")

epsilon = 1e-20
if error_final < epsilon:
    print("\nLa descomposición cumple: ||U^T A V - diag(mu)||_1 < tol")
else:
    print("\nLa descomposición No cumple la condición deseada.")
"""
