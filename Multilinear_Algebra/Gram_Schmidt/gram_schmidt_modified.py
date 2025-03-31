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
        Q[: ,0] = A[:, 0] / R[0,0] # Assign the entire normalized column vector as q_{1}

    for j in range(1, n):
        q_hat = A[:, j]

        """
        There is no longer an explicit summation. Instead of accumulating and then subtracting:
        Project onto Q[:, i]
        Immediately subtract that projection
        Move on to the next Q[:, i]

        This immediate update at each step helps maintain orthogonality
        """

        for i in range(j):
            R[i,j] = np.dot(q_hat, Q[:, i]) # Q[:, i] gets the i-th column vector
            q_hat = q_hat - R[i,j]*Q[:, i]

        R[j,j] = np.linalg.norm(q_hat)

        if R[j,j] < 1e-12: # Avoid rounding errors by not checking equality to zero
            print("Stop")
            print("Column vector is linearly dependent")
            return
        else:
            Q[:, j] = q_hat / R[j,j]

    return Q, R

A = np.random.rand(4, 3)

Q, R = gram_schmidt_modified(A)


print(f" A - QR = ", np.linalg.norm(A - Q@R))
