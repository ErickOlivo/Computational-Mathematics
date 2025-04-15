import numpy as np
def qr_iterative(A, tol=1e-12, max_iter=1000):
    """
    -------------------------------------------------------------------------------------------------------
    Allows the matrix A to be transformed into a Schur form where its wigenvalues appear on the diagonal.

    As the iterative process converges, the resulting matrix clearly dispays these values on the diagonal.

    When we decompose A = QR and then recombine A_{k+1} = RQ, we can show that A_{k+1} = Q^T A Q, since Q
    is orthogonal (Q^T Q = I). This means A_{k+1} is similar to A, thus they share the same eigenvaluess.

    This interative process "pushes" the eigenvalues to the diagonal.

    By Schur's theorem, if A is real, it guarantees that A is orthogonally similar to  an almost triangular
    matrix: Q^T A Q = T

    -------------------------------------------------------------------------------------------------------
    """
    n = A.shape[0]
    U = np.eye(n)    # Accumulator for orthogonal transformations

    iteration_count = 0

    """
    Continues until A_k converges to a triangular form (real) or quasi-block-triangular (if there are complex values)

    The diagonal entries of the converged matrix (final A_k) are the aproximations of the eigenvalues
    """
    while iteration_count < max_iter:
        # Step 1: Perform the QR decomposition on the current matrix
        Q, R = np.linalg.qr(A)

        U = np.dot(U, Q) # Accumulate the transformation into U (to build the required orthogonal matrix)

        # Step 2: Update the matrix A = RQ
        A = np.dot(R, Q)

        # Step 3: Evaluate the stopping criterion

        D = np.diag(np.diag(A)) # Extract the diagonal part of A

        # Compute the L1 norm of the off-diagonal elements
        error = np.sum(np.abs(A - D)) # Determines when the off-diagonal portion is small enough
                                      # The idea is that when this sum is very small,
                                      # we can say the off-diagonal elements are nearly zero.
        if error < tol:
            break

        iteration_count += 1

    return U, A # Upon exiting, A is almost diagonal and U holds the accumulated Q's


# ---------------
# Main code
# ---------------

# -------------------------------------------------------------------------------
# Step A: Define the original matrix A
# -------------------------------------------------------------------------------
A = np.random.rand(4, 4)

# Make a copy for the QR process (since A is modified in-place)
A_copy = A.copy()

# -------------------------------------------------------------------------------
# Step B: Apply the iterative QR to get U and the near-diagonal matrix A_diag (T)
# -------------------------------------------------------------------------------
U, A_diag = qr_iterative(A_copy) # A_diag is T

# Print partial results
print("U matrix (accumulated Q's):")
print(U)
print("\nNear-diagonal (T):")
print(A_diag)

# Schur check: Reconstruct using U and T
A_reconstructed = U @ A_diag @U.conj().T

print(f"Schur reconstruction error: A - U T U^H = {np.linalg.norm(A - A_reconstructed)}")

# -------------------------------------------------------------------------------
# Step C: Obtain V and the singular values (mu_i) via QR iteration on A^T A
# -------------------------------------------------------------------------------


"""
Obtain a matrix V and the singular values mu_i such that U^T A V = diag(mu_i), where the eigenvalues
(or, in the context of the SVD, the squares of the singular values) appear on the diagonal.

The singular values are obtained as u_i = sqrt(lambda_i), with lambda_i being the eigenvalues of A^T A
"""


# Define B = A^T A (or A.conj().T @ A for the complex case)
"""
In the real case, using B = A.T @ A and B = A.conj().T @ A is equivalent,
since the complex conjugate has no effect on real numbers.

To make the same code work correctly for the complex case as well, use A.conj().T @ A,
which is the proper way to compute A* A
"""

A_star = A.conj().T # Works for real or complex (complex-conjugate transpose)
B = A_star @ A

# Apply the same qr_iterative function on B to get V (the Q accumulation)
# and the near-diagonal form B_approx
V, B_approx = qr_iterative(B)

# The eigenvalues of B are approximated on the diagonal of B_approx
lambda_vals = np.diag(B_approx)

# Compute A's singular values: mu_i = sqrt(lambda_i), ensuring non-negativity
# np.maximum(lambda_vals, 0) compares each lambda_i with 0 and returns the greater of the two,
# so if lambda_i is slightly negative due to numerical or rounding errors, 0 will be used instead.
# This ensures that the square root is only taken of non-negative numbers

mu = np.sqrt(np.maximum(lambda_vals, 0.0))

print("\nSingular values (mu):")
print(mu)
print("\nMatrix V (obtained from QR on A^T A):")
print(V)



# Build the U matrix
"""
The U returned after calling qr_iterative(A) comes from performing QR on A itself and accumulating the Q factors.
This is the U that triangularizes A in the Schur form—it is not the U from the singular value decomposition (SVD).

The approximation is made by diagonalizing B = A^T A. With the V obtained from qr_iterative(B),
we already have the right-hand side of the SVD.

Now, we need to construct U using U_i = A @ v_i / u_i for each column v_i,
where v_i is the i-th column of V and u_i is the corresponding singular value.

The U obtained from triangularizing A (Schur form) is focused on turning A into a triangular matrix (i.e., eigenvalue-oriented),
whereas the U we are about to compute aims to diagonalize A in terms of its singular values.
"""

# -------------------------------------------------------------------------------
# Step D: Construct the U from the SVD, using A and V
# -------------------------------------------------------------------------------

U_cols = []
n = A.shape[0]

for i in range(n):
    """
    We can build U because applying A to v_i already gives the direction of u_i
    """
    if np.abs(mu[i]) < 1e-14:
        # If mu[i] is ~0, then A v_i=0 => v_i is in A's nullspace
        # The nullspace (or kernel) of A is the set of vectors x such that A x = 0.
        # Use a canonical basis vector e_i
        col_i = np.zeros(n, dtype=A.dtype)
        col_i[i] = 1.0
    else:
        # Build column i dividing (A v_i) by mu[i]
        v_i = V[:, i]
        col_i = (A @ v_i) / mu[i]

    U_cols.append(col_i)

# Convert the list of columns into a matrix
U_temp = np.column_stack(U_cols)

# Orthonormalize U_temp (QR) for greater numerical stability

U_svd, _ = np.linalg.qr(U_temp) # In theory, if each column was constructed as u_i = A @ v_i / mu, then U should also be orthonormal,
                                # since each v_i is orthonormal. However, due to rounding errors, the columns of U_temp may not be perfectly orthogonal.
                                # Orthonormal (or unitary) matrix; the columns of U_svd approximate the left singular vectors of A


# -------------------------------------------------------------------------------
# Step F: Validate the decomposition
# -------------------------------------------------------------------------------

"""
Once we have U_svd, mu, and V, we check that ||U^T A V - diag(mu)||_1 < epsilon
"""
# Build diag(mu)
Sigma = np.diag(mu)

# Compute the difference
M = (U_svd.T @ A) @ V # M is the full transformation of A using the orthogonal (or unitary) matrices U and V

"""
There is a sign ambiguity in the construction of the U and V factors, which does not affect the validity of the decomposition.

In the SVD, by definition, the singular values u_i must be >= 0.

It's enough to flip the sign of the i-th column to ensure that the diagonal entry u_i is positive.

Sometimes, (U^T A V) produces -u_i on the diagonal instead of +u_i, which increases the difference from diag(mu),
artificially inflating the 1-norm.
"""

# Force the diagonal of M to be positive
# This sign flip does not affect the orthonormality of U or V, since (-1)⋅(column) remains orthonormal,
# and it ensures that the diagonal is exactly diag(u_i) >= 0

for i in range(n):
    if M[i, i] < 0:
        M[:, i] = -M[:, i]
        U_svd[:, i] = -U_svd[:, i]

                                    # Now M should have diagonal >= 0


# ---- LEFT & RIGHT SINGULAR VECTORS ---------------------------------
print("\nLeft singular vectors (U_svd):")
print(U_svd)

print("\nRight singular vectors (V):")
print(V)
# -------------------------------------------------------------------------------------


diff = M - Sigma

"""
diff = M - Sigma


"""
The 1-norm of a matrix sums the absolute values of each column separately.
The 1-norm is the max of those column sums.
"""

err_1 = np.max(np.sum(np.abs(diff), axis=0))  # axis=0 sums the elements column-wise (i.e., vertically across each column)
                                              # np.max(...) takes the largest of those values


print("\nM = U^T A V:")
print(M)
print("\nDiagonal matrix:")
print(np.diag(mu))
print(f"\nFinal error in the 1-norm (||U^T A V - diag(mu)||_1): {err_1}")

epsilon = 1e-12
if err_1 < epsilon:
    print("\nThe decomposition satisfies the condition: ||U^T A V - diag(μ)||_1 < tol")
else:
    print("\nThe decomposition does NOT satisfy the desired condition.")
