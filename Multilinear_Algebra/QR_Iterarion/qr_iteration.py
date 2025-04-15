import numpy as np
def qr_iterative(A, tol=1e-12, max_iter=1000):
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

        """
        Este error mide cuánto de la matriz A sigue siendo "no diagonal"
        Cuando eso es muy pequeño, asumimos que A_k ya está casi diagonal, y los autovalores están en la diagonal
        """

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
"""
En el caso real, usar B = A.T @ A y usar B = A.conj().T @ A es equivalente, porque la conjugada de números reales no cambia nada.
Para que el mismo código funcione también para el caso complejo, usar A.conj().T @ A, que es la manera correcta de obtener A* A
"""
A_star = A.conj().T # Works for real or complex (complex-conjugate transpose)
B = A_star @ A

# Aplicamos la misma función QR iterativo a B para obtener V (acumulador de Q's para B) y la forma casi diagonal B_diag
V, B_approx = qr_iterative(B) # Matriz ortogonal V acumulada y una forma casi diagonal de B

# Los autovalores de B se aproximan a la diagonal de B_diag
# Se extraen de B_diag, y no de A_diag ya que que los autovalores de A no se relacionan de manera directa con los valores signulares
lambda_vals = np.diag(B_approx)

# Calcular los valores singulares de A: mu_i = sqrt(lambda_i) (asegurándonos de que sean no negativos)
# np.maximun(lambda_vals, 0) compara cada valor de lambda_i con 0 y devuelve el mayor de ambos, de esa manera si por errores numéricos y de redondeo lambda_i es ligeramente negativo, se usará 0 en su lugar.
# Así se asegura estar calculando la raíz cuadrada de números no negativos
mu = np.sqrt(np.maximum(lambda_vals, 0.0))

print("\nValores singulares (mu):")
print(mu)
print("\nMatriz V (obtenida a partir del QR iterativo aplicado a A^T A):")
print(V)



# Construir la matriz U
"""
La U que se había devuelto trans llamar a qr_iterative(A) proviene de hacer QR sobre la propia A y acumular los factores.
Esa es la U que diagonaliza o triangulairza A en la forma de Schur, no la U de la descomposición en valores singulares.

La aproximación se hace diagonalizando B = A^T A, con la V que se obtuvo de qr_iterative(B), ya se tiene la parte derecha de la SVD.

Ahora se requiere construir U mediante U_i = A v_i / u_i para cada olumna v_i, donde v_i es la i-ésima columna de V y u_i es el valor singular correspondiente.

La U obtenida al triangularizar A (Schur) está orientada a que A se vuelva triangular (enfocada a autovalores)
y la que se obtendrá ahora a que A se diagonalice respecto de sus valores singulares
"""
# Paso D: Construir la U de la SVD, usando A y V
U_cols = []
n = A.shape[0]

for i in range(n):
    """
    Se puede contruir U porque aplicar A a v_i ya te da la dirección de u_i
    """
    if np.abs(mu[i]) < 1e-14:
        # Si mu[i] es aproximandamente 0, elegimos un vector ortonormal cualquiera
        # Si mu = 0, entonces Av_i = 0. No se puede dividir entre cero, pero se sabe que v_i está en el núcleo (nullspace) de A
        # Así que u_i puede ser cualquier vectr ortonormal que complete la base
        # Un vector canónico e_i
        col_i = np.zeros(n, dtype=A.dtype)
        col_i[i] = 1.0
    else:
        # col_i = (A @ v_i) / mu_i
        v_i = V[:, i]
        col_i = (A @ v_i) / mu[i]

    U_cols.append(col_i)

# Convertir lista de columnas en matriz
U_temp = np.column_stack(U_cols)

# Ortonormalizar U_temp (QR) para mayor estabilidad numérica
# En teoría si ya se construyó cada columna como u_i = Av_i / mu, U debería ser ortonormal también, ya que cada v_i lo es, al haber errores de rendondeo, por lo que las columnas de U_temp pueden no ser perfectamente ortogonales
U_svd, _ = np.linalg.qr(U_temp) # Matriz ortonormal (o unitaria), las columnas de U_svd aproximan los vectores isngulares izquierdos de A


# Paso F: Validad descompoisicón
"""
Una vez se tiene U_svd, mu y V, se verifica que
||U^T A V - diag(u)||_1 < epsilon
"""
# Construir diag(mu)
Sigma = np.diag(mu)

# Calcular la diferencia
M = (U_svd.T @ A) @ V # M es la tranformación completa de A usando las matrices ortogonales (o unitarias) U y V,

"""
Existe una libertad de signo en la constucción de los factores U y V, sin alterar la validez de la descomposición

En la SVD, por definición u_i >= 0

Basta con cambiar el signo de la columna i-ésima, haciendo que la diagonal sea u_i positiva

A veces (U^T A V) produce -u_i en la diagonal en lugar de +u_i. Haciendo que la diferencia con diag(mu) sea grande, inflando la norma-1
"""
# (Tras calcular M = U_svd.T @ A @ V), para forzar que la diagonal de U^T A V sea positiva
for i in range(n):
    if M[i, i] < 0:
        # Este sign flip no altera la ortonormalidad de U o V, pues (-1)⋅(columna) sigue siendo ortonormal,
        # garantiza que la diagonal sea exactamente diag(u_i) >= 0
        M[:, i] = -M[:, i]
        U_svd[:, i] = -U_svd[:, i]
# Ahora M debería tener diagonal >= 0

diff = M - Sigma # Verifico la descomposición, uso M para compararla con la matriz diagonal construida a partir de los valores singulares mu

"""
La norma-1 de una matriz suma los valores absolutos de cada columna por separado,
el resultado de la norma-1 es el mayor de todas esas sumas
"""
# np.max(...) toma el mayor de esos valores
err_1 = np.max(np.sum(np.abs(diff), axis=0)) # Suma cada columna, axis=0 suma los elementos por columna, es decir, suma verticalmente cada columna de la matriz

print("\nM = U^T A V:")
print(M)
print("\nMatriz diagonal:")
print(np.diag(mu))
print(f"\nError final (||U^T A V - diag(mu)||_1): {err_1}")

epsilon = 1e-12
if err_1 < epsilon:
    print("\nLa descomposición cumple: ||U^T A V - diag(μ)||_1 < tol")
else:
    print("\nLa descomposición No cumple la condición deseada.")
