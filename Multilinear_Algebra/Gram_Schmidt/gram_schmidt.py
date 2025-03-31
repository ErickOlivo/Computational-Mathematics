import numpy as np

def gram_schmidt(A):
    m, n = A.shape

    """
    This is not necessary, since linear dependence is detected in the loop

    rank = np.linalg.matrix_rank(A)
    if rank != n:
        print("Linearly dependent matrix")
        return
    """

    R = np.zeros((m,n)) # nxn to make it upper triangular and square

    Q = np.zeros((m,m)) # mxn to store the columns correctly


    R[0,0] = np.linalg.norm(A[:, 0])
    if R[0,0] == 0:
        print("First column vector is linearly dependent")
        return
    else:
        Q[:, 0] = A[:, 0] / R[0,0]

    for j in range(1, n):
        sum_projections = np.zeros(m)
        for i in range(j):
            R[i,j] = np.dot(A[:, j], Q[:, i])
            sum_projections += R[i,j] * Q[:, i]

            # Now we build a new vector by subtracting from x_j the projections onto the previous q_i vectors
            # Q_hat is a vector orthogonal to all previous q_i, but it is not yet normalized
            # We subtract from x_j all the parts in the direction of the previous vectors, keeping only the perpendicular component
            # Where (x_j, q_i) is the projection of x_j onto q_i


        q_hat = A[:, j] - (sum_projections)


        """
        Step 5
        """
        R[j,j] = np.linalg.norm(q_hat)


        """
        Step 6
        """
        if R[j,j] == 0:
            print("Stop")
            print("Vector is linearly dependent")
            return
        else:
            Q[:, j] = q_hat / R[j,j]

    return Q, R



A = np.random.rand(4, 3)
Q, R = gram_schmidt(A)


print(f"A - QR = ", np.linalg.norm(A - Q@R))
