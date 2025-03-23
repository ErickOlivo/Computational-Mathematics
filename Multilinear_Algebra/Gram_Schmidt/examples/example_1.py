import numpy as np
from ..gram_schmidt import gram_schmidt

# Definir una matriz de ejemplo
A = np.array([[1.0, 1.0],
              [1.0, 0.0],
              [0.0, 1.0]])

# Aplicar Gram-Schmidt
Q, R = gram_schmidt(A.copy())

# Mostrar resultados
print("Matriz ortonormal Q:")
for q in Q:
    print(q)

print("\nMatriz triangular superior R:")
print(R)
