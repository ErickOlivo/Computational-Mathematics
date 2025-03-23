import numpy as np
from ..qr_factorization import qr_factorization, power_method, schur_decomposition

# Define a sample matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
x0 = np.random.rand(A.shape[0])

# QR Factorization
Q, R = qr_factorization(A)
print("Q:\n", Q)
print("R:\n", R)

# Power Method
lambda_dominant, eigenvector_dominant = power_method(A, x0)
print(f"Dominant eigenvalue: {lambda_dominant}")
print(f"Corresponding eigenvector: {eigenvector_dominant}")

# Schur Decomposition
Q_schur, R_schur = schur_decomposition(A)
print("Unitary matrix Q:\n", Q_schur)
print("Upper triangular matrix R:\n", R_schur)

# Verification: Q^H A Q = R
R_verification = np.dot(Q_schur.T, np.dot(A, Q_schur))
print("R computed:\n", R_schur)
print("R verification (Q^H A Q = R):\n", R_verification)
