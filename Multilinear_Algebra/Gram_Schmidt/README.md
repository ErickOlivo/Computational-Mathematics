# Gram-Schmidt Orthogonalization

This project implements the classical Gram-Schmidt algorithm for orthogonalizing a set of vectors in Python using NumPy.

## Algorithm
The Gram-Schmidt process takes a set of linearly independent vectors and produces an orthonormal set of vectors that spans the same subspace.

## Usage
To use the algorithm, import the `gram_schmidt` function from `gram_schmidt.py`:

```python
from gram_schmidt import gram_schmidt

# Define your vectors as columns of a matrix
A = np.array([[1.0, 1.0],
              [1.0, 0.0],
              [0.0, 1.0]])

# Apply Gram-Schmidt
Q, R = gram_schmidt(A.copy())
