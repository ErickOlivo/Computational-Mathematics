import numpy as np

def householder(A):
    """
    Performs QR decomposition of a matrix A using Householder reflections.

    Instead of orthogonalizing columns using projections,
    this method reflects vectors to eliminate components below the diagonal,
    preserving orthogonality.

    When a vector x is reflected with respect to a hyperplane,
    a new vector x' is generated that is:
    - Symmetric to x with respect to the hyperplane
    - On the opposite side, at the same distance from the plane

    P = I - 2 w w^T
    If w is a column vector (m x 1), then w^T is a row vector (1 x m)

    w w^T results in an m x m matrix used to reflect any vector x in that space
    """

    X = A.copy()
    R = X.copy()

    n, m = X.shape

    # Check if columns are linearly independent
    rank = np.linalg.matrix_rank(A)
    if rank != m:
        print("Linearly dependent matrix")
        return

    W = np.zeros((n, m))    # Stores the reflection vectors w_k

    Q = np.zeros((n, m))

    for k in range(0, m):
        r_k = R[:, k].copy()

        # if k > 0:
            # r_k = R[:, k].copy()

        # Apply previous reflections P_1, P_2, ..., P_{k-1} to column x_k to obtain r_k
        for i in range(k):
            w_i = W[:, i]

            # r_k:= P_{k-1}...P_{1}x_k which is equivalent to:
            # P_i x = x - 2 w_i (w_i^T x)
            # This applies the reflection without explicitly constructing P_i
            r_k = r_k - 2 * w_i * (w_i.T @ r_k) # P = I - 2 w w^T


        # Step 4: Compute w_k

        # Extract subvector
        x = r_k[k:] # Subvector from position k to end

        # Compute beta (equation 1.21)
        """
        X[0] is r_k[k], i.e., the first element starting at position k
        x[1] would be r_k[k+1]

        np.linalg.norm(x) is ||r_k[k:]||_2
        """
        """
        The value beta is used to construct the vector w_k.
        It is adjusted according to the sign of the first component of x.
        """
        beta = np.sign(x[0]) * np.linalg.norm(x)

        # Compute vector z (equation 1.20)
        """
        z = [ 0, 0, 0, r_k[k] + beta, r_k[k+1], r_k[k+2], ... ]
            ↑      ↑
          i < k    i = k     (and i > k)

        """
        z = r_k.copy()          # for i > k
        z[:k] = 0               # for i < k, set zeros in all positions before k
        z[k] = r_k[k] + beta    # for i = k, modify value at position k in z by adding beta


        # Normalize z to obtain w_k
        w_k = z / np.linalg.norm(z)
        # Store w_k in matrix W
        W[:, k] = w_k


        # Step 5: Reflect the entire matrix R

        # R := R - 2 w_k (w_k.T R)
        """
        If w_k is a column vector (nx1)
        And v = w_k.T @ R is a row vector (1xm)
        np.outer(w_k, v) gives a matrix (nxm)
        """
        R = R - 2 * np.outer(w_k, w_k.T @ R) # We use the outer product to update R using w_k


        # Step 6: Compute q_k = P_1 P_2 ... P_k e_k
        q_k = np.zeros(n)       # Initialize zero vector of same length as rows in A
        q_k[k] = 1              # Standard basis vector e_k

        # Apply reflections in reverse order: P_k, P_{k-1}, ..., P_0
        """
        This is necessary because the reflections in P_k affect not only column k of the matrix,
        but also later columns.
        To obtain Q, we apply the transformations in reverse order from how they were calculated.
        """
        for i in reversed(range(k+1)):
            w_i = W[:, i]
            q_k = q_k - 2 * w_i * (w_i.T @ q_k)

        # Store q_k as column k of Q
        Q[:, k] = q_k

    R = R[:m, :] # Keep only the first m rows; R should have the same number of rows as A has columns
    return Q, R

A = np.random.rand(3, 2)

Q, R = householder(A)

print(f"A - QR = ", np.linalg.norm(A - Q@R))

print(f"Q^T Q = ", np.allclose(Q.T @ Q, np.eye(Q.shape[1])))


