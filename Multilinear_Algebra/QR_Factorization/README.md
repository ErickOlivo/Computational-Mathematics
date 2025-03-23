# QR Factorization and Related Algorithms

This project implements the following algorithms:
- **QR Factorization**: Decomposes a matrix into an orthogonal matrix (Q) and an upper triangular matrix (R).
- **Power Method**: Computes the dominant eigenvalue and eigenvector of a matrix.
- **Schur Decomposition**: Decomposes a matrix into a unitary matrix (Q) and an upper triangular matrix (R).

## Usage
To use the algorithms, import the functions from `qr_factorization.py`:

```python
from qr_factorization import qr_factorization, power_method, schur_decomposition

# Define your matrix
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# QR Factorization
Q, R = qr_factorization(A)

# Power Method
lambda_dominant, eigenvector_dominant = power_method(A, x0)

# Schur Decomposition
Q_schur, R_schur = schur_decomposition(A)
