import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import lu_factor
from scipy.linalg import lu_solve
import time

def gauss(A):
    """
    Gaussian elimination with backsubstitution on matrix A (list of lists)

    Returns upper triangular matrix A  

    Code credit: https://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/
    """
    n = len(A)

    for i in range(0, n):
        # Search for maximum in this column
        maxEl = abs(A[i][i])
        maxRow = i
        for k in range(i+1, n):
            if abs(A[k][i]) > maxEl:
                maxEl = abs(A[k][i])
                maxRow = k

        # Swap maximum row with current row (column by column)
        for k in range(i, n+1):
            tmp = A[maxRow][k]
            A[maxRow][k] = A[i][k]
            A[i][k] = tmp

        # Make all rows below this one 0 in current column
        for k in range(i+1, n):
            c = -A[k][i]/A[i][i]
            for j in range(i, n+1):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += c * A[i][j]

    return A

def solve_gauss_elimination(A):
    """
    Solve equation Ax=b for a matrix A
    """
    # calculate upper triangular matrix
    A = gauss(A) 
    n = len(A)

    x = [0 for i in range(n)]
    for i in range(n-1, -1, -1):
        x[i] = A[i][n]/A[i][i]
        for k in range(i-1, -1, -1):
            A[k][n] -= A[k][i] * x[i]

    return x

def GaussJordan(A):
    """
    Gauss Jordan elimination matrix A (list of lists)
    """    
    # find the upper triangular matrix
    A = np.asarray(gauss(A))

    m,n = A.shape

    for i in reversed(range(m)):
        # divide by pivot
        A[i,:] /= A[i,i]

        # then reduce numbers behind the pivot to 0
        counter = 1
        for j in range(i + 1, n-1):
            # by subtracting the corresponding pivot row multiplied by a factor
            a = A[i,j]/ A[i+counter, j]
            A[i,:] -= a*A[i+counter, :]
            counter += 1

    return A

if __name__ == "__main__":
    n = 6

    A = [[3, 8, 1, -12, -4, 2],
         [1, 0, 0, -1, 0, 0],
         [4, 4, 3, -40, -3, 1],
         [0, 2, 1, -3, -2, 0],
         [0, 1, 0, -12, 0, 0]]


    # Calculate solution
    t = time.time()
    x = solve_gauss_elimination(A)
    t_ge = time.time() - t
    print ("Result Gauss elimination:\t", x)
    print (f"Took {t_ge} seconds")


    """
    (Correct) result:
    # [0.08333333333333348, 1.0000000000000004, 1.750000000000001, 0.08333333333333337, 1.750000000000001]
    Which is
    # [1/12, 1, 7/4, 1/12, 7/4] wrt x_6
    
    (see also https://matrix.reshish.com/gauss-jordanElimination.php)
    """

    # So we multiply by 12 to get [ 1., 12., 21.,  1., 21.]
    x = np.asarray(x)*12

    A = [[3, 8, 1, -12, -4, 2],
     [1, 0, 0, -1, 0, 0],
     [4, 4, 3, -40, -3, 1],
     [0, 2, 1, -3, -2, 0],
     [0, 1, 0, -12, 0, 0]]

    t = time.time()
    A_gj = GaussJordan(A)
    t_gj = time.time() - t
    print ("Matrix GaussJordan elimination:\n", A_gj)
    print (f"Took {t_gj} seconds")
    """
    Same answer
    """

    # LU solver
    t = time.time()
    A = np.asarray(A)
    x_lu = lu_solve( lu_factor(A[:,:-1]), A[:,-1] )
    t_lu = time.time() - t
    print ("Result LU decomp:\t", x_lu)
    print (f"Took {t_lu} seconds")
    """
    Same answer
    """







