import numpy as np

def gram_schmidt_modified(A):
    m, n = A.shape

    Q = np.zeros((m,n))
    R = np.zeros((n,n))

    R[0,0] = np.linalg.norm(A[:, 0])

    if R[0,0] < 1e-12:
        print("Stop")
        print("First vector is linearly dependent")
        return

    else:
        Q[: ,0] = A[:, 0] / R[0,0] # Assign the full normalized column vector as q_{1}


    for j in range(1, n):
        q_hat = A[:, j]

        """
        There is no longer an explicit summation; instead of accumulating and then subtracting:
        Project onto Q[:, i]
        Immediately subtract that projection
        Proceed to the next Q[:, i]

        This immediate update at each step helps maintain orthogonality
        """
        for i in range(j):
            R[i,j] = np.dot(q_hat, Q[:, i]) # Q[:, i] is the i-th column vector
            q_hat = q_hat - R[i,j]*Q[:, i]

        R[j,j] = np.linalg.norm(q_hat)

        if R[j,j] < 1e-12: # Avoid exact comparison with zero due to rounding
            print("Stop")
            print("Column vector is linearly dependent")
            return
        else:
            Q[:, j] = q_hat / R[j,j]

    return Q, R


def power_method(A, v, tol = 1e-3, max_iter = 10000):

    lambda0 = 0
    convyn = 0 # 0: did not converge, 1: converged
    for i in range(max_iter):

        v = np.dot(A, v)
        v = v /np.linalg.norm(v)

        lambda1 = np.dot(np.transpose(v), np.dot(A, v))[0, 0]
        L = np.abs(lambda0 - lambda1)

        if L < tol:
            convyn = 1
            break
        lambda0 = lambda1

    if i == max_iter-1:
        print("Convergence not reached within the maximum number of iterations")
    return convyn, lambda1, v


def schur_recursive(A, tol=1e-3):
    n = A.shape[0] # Only row count is used, since A is square

    if n == 1:
        return np.eye(1), A.copy()

    # Step 1: Obtain dominant eigenvalue and eigenvector
    v0 = np.random.rand(n, 1)
    _, lambda1, u = power_method(A, v0, tol=tol)

    # Step 2: Complete u to an orthonormal basis U = [u, V]
    u = u.reshape(-1, 1)
    V = np.eye(n)[:, 1:]
    U_aux = np.hstack([u, V])
    Q_U, _ = gram_schmidt_modified(U_aux)
    U = Q_U

    # Step 3: Transform A -> B = U^H A U (Change of basis), yielding a new similar matrix expressed in the new basis (U)
    B = U.conj().T @ A @ U # The first eigenvalue will appear in B[0, 0]

    # Step 4: Extract (n-1)x(n-1) submatrix from bottom-right
    B_sub = B[1:, 1:]

    # Step 5: Apply recursion to B_sub
    # B becomes more triangular at each step
    Q1, R1 = schur_recursive(B_sub)

    # Step 6: Build Q_hat
    Q_hat = np.eye(n)
    Q_hat[1:,1:] = Q1

    # Step 7: R = Q_hat^H B Q_hat
    R = Q_hat.conj().T @ B @ Q_hat

    # Step 8: Q = U Q_hat
    Q = U @ Q_hat

    return Q, R

A = np.random.rand(4, 4)

Q, R = schur_recursive(A)

A_reconstructed = Q @ R @ Q.conj().T


print(f"A - Q R Q^H: {np.linalg.norm(A - A_reconstructed)}")

print(f"Q^H Q - I: {np.linalg.norm(Q.conj().T @ Q - np.eye(A.shape[0]))}")


