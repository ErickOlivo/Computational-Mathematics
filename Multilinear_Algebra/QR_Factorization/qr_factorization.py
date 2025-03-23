import numpy as np

def qr_factorization(A):
    """
    Computes the QR factorization of a matrix A using the Gram-Schmidt process.

    Parameters
    ----------
    A : np.ndarray
        Input matrix of shape (m, n).

    Returns
    -------
    Q : np.ndarray
        Orthogonal matrix of shape (m, n).
    R : np.ndarray
        Upper triangular matrix of shape (n, n).
    """
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for i in range(n):
        v = A[:, i]
        for j in range(i):
            R[j, i] = np.dot(Q[:, j], A[:, i])
            v = v - R[j, i] * Q[:, j]
        R[i, i] = np.linalg.norm(v)
        Q[:, i] = v / R[i, i]

    return Q, R

def power_method(A, x0):
    """
    Computes the dominant eigenvalue and eigenvector using the Power Method.

    Parameters
    ----------
    A : np.ndarray
        Input matrix of shape (n, n).
    x0 : np.ndarray
        Initial guess vector of shape (n,).

    Returns
    -------
    lambda_dominant : float
        Dominant eigenvalue.
    eigenvector_dominant : np.ndarray
        Corresponding eigenvector.
    """
    tol = 0.00001
    kmax = 100

    x_k_anterior = x0 / np.linalg.norm(x0)
    lambda_k_anterior = 0

    for k in range(kmax):
        x_k_actual = np.dot(A, x_k_anterior)
        x_k_actual = x_k_actual / np.linalg.norm(x_k_actual)
        lambda_k_actual = np.dot(x_k_actual.T, np.dot(A, x_k_actual))  # Rayleigh quotient

        if np.abs(lambda_k_actual - lambda_k_anterior) < tol:
            break

        x_k_anterior = x_k_actual
        lambda_k_anterior = lambda_k_actual

    return lambda_k_anterior, x_k_anterior

def schur_decomposition(A):
    """
    Computes the Schur decomposition of a matrix A using QR factorization.

    Parameters
    ----------
    A : np.ndarray
        Input matrix of shape (n, n).

    Returns
    -------
    Q : np.ndarray
        Unitary matrix of shape (n, n).
    R : np.ndarray
        Upper triangular matrix of shape (n, n).
    """
    tol = 0.00001
    kmax = 100

    n = A.shape[0]  # n is the size of the matrix A
    Q_total = np.eye(n)  # initialize Q as the identity matrix
    Ak = np.copy(A)

    for k in range(kmax):
        # Compute QR factorization of Ak
        Q, R = qr_factorization(Ak)
        Ak = np.dot(R, Q)
        Q_total = np.dot(Q_total, Q)

        # Check convergence
        off_diagonal = np.sqrt(np.sum(np.tril(Ak, -1) ** 2))  # norm of lower triangular part
        if off_diagonal < tol:
            break

    return Q_total, Ak
