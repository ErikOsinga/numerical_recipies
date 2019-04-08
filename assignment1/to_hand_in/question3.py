import numpy as np
import matplotlib.pyplot as plt

from some_routines import linspace, logspace

import question2 as q2

def neglogL(a,b,c,xi,sumlogxi):
    """
    Returns the negative log likelihood for a set of N i.i.d. 
    realizations x_i, given parameters a,b,c
    
    Takes the sum of the logarithm of xi values as well
    because this can be calculated once upfront for a single file.
    This removes the unnecessary calculation of this value 
    every time we evaluate neglogL for certain a,b,c
    """
    N = len(xi)
    
    if (not(1.1 < a <2.5) or not(0.5 < b < 2) or not(1.5 < c < 4)):
        # a b or c is not in required bound, return high value
        return 1e6
    
    # We use the trilinear interpolator to approximate A(a,b,c)
    A = q2.linInterp([a,b,c])
    
    second_term = sumlogxi
    second_term -= N*np.log(b)
    
    return -1* ( N*np.log(A) + (a-1)*second_term - b**(-c) * np.sum(xi**c) )

def calc_centroid(vectors):
    """
    Given an array 'vectors' of shape (len(points),Ndim)
    calculate the centroid (i.e., mean over every dimension)
    """
    centroid = np.empty(vectors.shape[1])
    for i in range(vectors.shape[1]):
        centroid[i] = np.mean(vectors[:,i])
        
    return centroid

def selection_argsort(arr):
    """
    Return indices that would sort array. Needed for downhill simplex function. 
    Since the number of points is usually low, we can just use selection sort.
    """
    N = len(arr)
    indices = list(range(0,N))
    # for every position in the array     
    for i in range(0,N-1):
        # Find next smallest element     
        imin = i
        for j in range(i+1, N):
            if arr[indices[j]] < arr[indices[imin]]:
                imin = j
        # put it in the correct position by swapping with current i
        if imin != i:
            indices[imin], indices[i] = indices[i], indices[imin]
    return indices

def downhill_simplexND(func,start,delta,tol=2e-10,maxIter=1000):
    """
    Find minimum of N-D function using the downhill simplex method
    
    func -- function to minimize (has to take x,y,.., as arguments)
    start -- vector of length N with the initial starting point
    delta -- guess for the characteristic length scale of the problem
    
    Returns
    x0 -- list containing the best guess for the minimum of the function
    vertices -- list of all vertices around the minimum
    it -- number of iterations done
    """
    N = len(start) # dimensionality
    it = 0
    
    # A vertex in N-D has N+1 points
    # First construct the other N points
    vertices = [start]
    for i in range(0,N):
        basis_vector = np.zeros(N)
        basis_vector[i] = 1
        vertices.append(start + basis_vector*delta)
        
    vertices = np.array(vertices) # array of shape (N+1,N)
    
    for _ in range(maxIter):
        it += 1
        
        # Order the points such that x0 is the minimum, xN is maximum
        order = selection_argsort([func(*vertice) for vertice in vertices])
        vertices = vertices[order]

        # Calculate the centroid of the first N points
        centroid = calc_centroid(vertices[:N])

        # Check if fractional range in func is within target acc
        fracrange = 2*abs(func(*vertices[-1]) - func(*vertices[0])) / (
                    abs(func(*vertices[-1]) + func(*vertices[0])) )
        if fracrange <= tol:
            # best guess x0, return all vertices too, and num iterations
            return vertices[0], vertices, it
        
        # Otherwise, propose a new point by reflecting xN
        x_try = 2*centroid - vertices[-1]

        # There are now four distinct possibilities:
        if func(*vertices[0]) <= func(*x_try) < func(*vertices[-1]):
            # new point is better, but not the best. We accept it
            vertices[-1] = x_try 
        elif func(*x_try) < func(*vertices[0]):
            # new point is the best, expand further in this direction
            x_exp = 2*x_try - centroid
            if func(*x_exp) < func(*x_try):
                # then expansion was even better
                vertices[-1] = x_exp
            else:
                # expansion did not work
                vertices[-1] = x_try
        else: # We know now that f(x_try) was worse than func(x_{N-1})
            # Propose a new point by contracting instead of reflecting
            x_try = 0.5*(centroid+vertices[-1])
            if func(*x_try) < func(*vertices[-1]):
                # Accept the contracted point
                vertices[-1] = x_try
            else:
                # All options were apparently bad, just zoom in on best
                for i in range(1,N+1):
                    vertices[i] = 0.5*(centroid + vertices[i])

    return vertices[0], vertices, it

def read_in_halos(filename):
    """
    Read in the data for exercise 3, per filename.
    
    First number in the file on line 4 is always the number of halos.
    Then we find hash symbols to indicate a new halo, and then we find 
    coordinates of satellites (if any): x, phi and theta.
    
    Returns the number of haloes in the datafile
    and the satellite positions x
    """
    
    # List of satellite positions x, don't care in which halo they are
    all_satellites = []
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            if i == 3:
                num_halos = int(line.strip('\n'))
            
            # i = 4 is always the first halo.
            if i > 4:
                if '#' in line: # Next halo 
                    pass
                else: # append satellite positions
                    coordinates = line.strip('\n').split('  ')
                    # Save only x position, cast to float
                    coordinates = float(coordinates[0])
                    all_satellites.append(coordinates)    
    
    return num_halos, all_satellites
            
def find_abc(i):
    """
    Find a,b,c that maximize the likelihood for mass file 'i'
    i has to be in [11,12,13,14,15]
    """
    num_halos, all_satellites = read_in_halos(f'./satgals_m{i}.txt')
        
    # Calculate this upfront for efficiency
    sumlogxi = np.sum(np.log(all_satellites))  

    # Function to minimize
    minfunc = lambda x, y, z: neglogL(x,y,z,all_satellites,sumlogxi)
    
    # Downhill simplex method
    
    initial_guess = [1.5, 0.7, 2.7] # from eyeballing the data
    best, vertices, it = downhill_simplexND(minfunc, initial_guess
                                        , 0.01,tol=1e-15,maxIter=500)
    return all_satellites, best, it
    

# We minimize with the downhill simplex method
all_best = [] # save best (a,b,c) for every mass file
for i in [11,12,13,14,15]:
    all_satellites, best, it = find_abc(i)
    all_best.append(best)
    print (f"Mass file m{i}")
    print (f"Best guess for a,b,c after {it} iterations: {best}")


all_best = np.asarray(all_best)

plt.scatter(range(11,16),all_best[:,0])
plt.title("$a$ as function of halo mass")
plt.xlabel('Halo mass')
plt.ylabel('a')
plt.savefig('./plots/q3b1.png')
plt.close()

plt.scatter(range(11,16),all_best[:,1])
plt.title("$b$ as function of halo mass")
plt.xlabel('Halo mass')
plt.ylabel('b')
plt.savefig('./plots/q3b2.png')
plt.close()

plt.scatter(range(11,16),all_best[:,2])
plt.title("$c$ as function of halo mass")
plt.xlabel('Halo mass')
plt.ylabel('c')
plt.savefig('./plots/q3b3.png')
plt.close()