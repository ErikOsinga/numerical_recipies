import numpy as np

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
seed = 19231923
print (f"User seed is set to {seed}")


class RandomGenerator(object):
    """
    Random generator should be an object because it maintains
    internal state between calls.
    """
    def __init__(self, seed):
        # make sure the everyhing is an unsigned 64 bit integer
        dtyp = np.uint64
        # the seed for the LGC
        self.X1 = dtyp(seed)
        # the seed for the XORshift
        self.X2 = dtyp(seed)
        
        self.max_value = dtyp(2**64 - 1)
        
        # LCG values from Numerical Recipies
        self.a = dtyp(1664525)
        self.c = dtyp(1013904223)
        self.m = dtyp(2**32)
        
        # 64 bit XOR shift values from Numerical Recipies
        self.a1, self.a2, self.a3 = dtyp(21), dtyp(35), dtyp(4)
        
    def lincongen(self, X):    
        return (self.a*X+self.c) % self.m

    def XORshift64(self, X):
        if X == 0:
            raise ValueError("Seed cannot be zero")
        X = X ^ (X >> self.a1)
        X = X ^ (X << self.a2)
        X = X ^ (X >> self.a3)
        
        return X
    
    def get_randomnumber(self):
        """
        Combine LCG and XORshift to produce random float 
        between 0 and 1
        """
        self.X1 = self.lincongen(self.X1)
        self.X2 = self.XORshift64(self.X2)
        
        # output is XOR of these numbers
        
        return (self.X1^self.X2)/self.max_value
    
RNGESUS = RandomGenerator(seed=seed)

# Randomly generate a,b,c within asked bounds
a = RNGESUS.get_randomnumber()*(2.5-1.1) + 1.1
b = RNGESUS.get_randomnumber()*(2.0-0.5) + 0.5
c = RNGESUS.get_randomnumber()*(4-1.5) + 1.5

print (f'a, b, c = {a,b,c}')

all_randnum = []
for i in range(int(1e6)):
    all_randnum.append(RNGESUS.get_randomnumber())
plt.hist(all_randnum,bins=np.linspace(0,1,21),edgecolor='black')
plt.title(f'Histogram of 1 million random numbers')
plt.xlabel('Value')
plt.ylabel('Counts')
#plt.show();
plt.close()



def numbdensprofile(x, a, b, c, A=1, Nsat = 100, spherical=False):
    """
    Returns the number density profile from the assignment
    
    a controls small-scale slope
    b contols transition scale
    c controls steepness of exponential dropoff
    spherical -- True to multiply by x**2 for the spherical integral
    """
    if spherical:
        power = 1
    else:
        power = 3
        
    return A*Nsat * (x/b)**(a-power) * np.exp( -(x/b)**c )
    
def romberg(func, lbound, ubound, order=6):
    """
    Calculate the integral of a function using Romberg's method
    with equal spaced abscissae
    
    func -- function which gives the y values
    lbound -- lower bound of integral
    ubound -- upper bound of integral
    N      -- number of abscissae
    
    efficiency:
    http://homen.vsb.cz/~lud0016/NM/Lecture_Notes_10-Romberg_Integration.pdf
    
    Returns 
     Value of the integral
     Error estimate
     
    The error estimate is given as the difference between last 2 orders
    """
    
    # for saving S_i,j's
    all_S = np.zeros((order,order))
    
    i = 0
    delta_x = (ubound-lbound)
    points = np.linspace(lbound,ubound,2**i+1)
    integral = delta_x/2 * np.sum(func(points))
    all_S[0,0] = integral
    
    # Then calculate the first column (S_{i,0})
    for i in range(1,order):
        delta_x /= 2
        # add points in the middle
        points = np.linspace(lbound,ubound,2**i+1)
        # add new points to the integral (om en om, starting from 1)
        integral = 0.5*integral + delta_x * np.sum(func(points[1::2]))
        
        all_S[i,0] = integral
    
    for j in range(1,order): # columns
        for i in range(j,order): # rows
            #print (i,j)
            #print (f'{4**j}*S{i},{j-1} - S{i-1},{j-1} / {4**j} - 1' )
            all_S[i,j] = (4**j*all_S[i,j-1] - all_S[i-1,j-1]) / (
                           4**j - 1)

    # compare this for the error function with the slides. 
    # print (all_S)
    
    return all_S[order-1,order-1], (all_S[order-1,order-1]-all_S[order-1,order-2])
        


# integral is only a function of R so add the prefactor manually
prefactor = 4*np.pi # integral over theta and phi
Nsat = 100
print ("For the following ")
print (f'a, b, c = {a,b,c}')
# print ("Romberg integration gives:")
integ, error = romberg(lambda x: numbdensprofile(x, a, b, c, Nsat=Nsat
                        , spherical=True) , 0, 5,order=10)
integ *= prefactor
# Normalize such that the integral produces <Nsat>
A = Nsat/integ
print (f"We find A = {A}")

integ, error = romberg(lambda x: numbdensprofile(x, a, b, c, A, Nsat=Nsat
                        , spherical=True) , 0, 5,order=10)
print ("Integral after changing A:", integ*prefactor)


def pdfRadii(x):
    """
    pdf for question 2d
    """
    # Set Nsat = 1 so we effectively divide by Nsat
    return ( numbdensprofile(x,a,b,c,A,Nsat=1,spherical=True) * 4*np.pi)
    
# Rejection sampling
def rejection_sampling(RNG, pdf, x_begin, x_end, num_points=500):
    """
    Sample num_points from the PDF by rejection sampling
    
    RNG -- random number generator that generates number \in [0,1]
    pdf -- pdf of the function to sample from: y(x)
    num_points -- amount of points to sample
    x_begin, x_end -- begin and endpoint of uniform dist sample
    """
        
    all_x = []
    factor = x_end-x_begin
    while len(all_x) < num_points:
        x = RNG()*(factor)+x_begin 
        y = RNG()
        if y <= pdf(x):
            all_x.append(x)
    
    return all_x
    
RNG = RNGESUS.get_randomnumber # random number generator func

def inverse_transform_sample(RNG, invcdf, num_points=500):
    """
    Sample num_points from the CDF with inverse transform sampling
    
    RNG -- random number generator that generates number \in [0,1]
    invcdf -- inverse CDF of the function to sample from
    num_points -- amount of points to sample
    """   
    
    all_x = []
    for _ in range(num_points):
        u = RNG()
        x = invcdf(u)
        all_x.append(x)
        
    return all_x
    
def invcdf(u):
    """Inverse CDF of \theta"""
    return np.arccos(1-2*u)
    
    
def output_N_satellites(RNG, N):
    """
    Generate N satellites following the number density profile
    given in the assignment
    
    We assume 
    0<=x<=5, 0<=\phi<2\pi, 0<= theta <= \pi
    
    Inputs
    RNG -- a function that will generate x ~ unif[0,1] when called
    N -- number of satellites
    """
    
    # Generate x with rejection sampling,
    # because we cannot invert or integrate the PDF
    all_x = rejection_sampling(RNG, pdfRadii, 0, 5, N) # x = r/rvir

    # Generate theta with inverse transform sampling
    # because integrating and inverting is very easy
    all_theta = inverse_transform_sample(RNG, invcdf, N)
    
    # Generate phi with just uniform numbers between 0 and 2pi
    all_phi = []
    for i in range(N):
        all_phi.append(RNG())
    all_phi = np.array(all_phi)*2*np.pi

    return all_x, all_theta, all_phi
    
# Output the positions for 100 such satellites
all_x, all_theta, all_phi = output_N_satellites(RNG,N=100)
# print (all_x,all_phi,all_theta)

# A thousand halos with 100 satellites
all_all_x, all_all_theta, all_all_phi = [], [], []
for i in range(1000):
    all_x, all_theta, all_phi = output_N_satellites(RNG,N=100)
    all_all_x.append(all_x)
    all_all_theta.append(all_theta)
    all_all_phi.append(all_phi)
    
all_all_x = np.asarray(all_all_x)

xs = np.linspace(1e-4,5,100)
# Generate 20 logarithmically equal spaced bins between 1e-4 and 5
bins = np.logspace(np.log10(1e-4),np.log10(5),21)

##### This part is for testing
nperbin, _, _ = plt.hist(all_all_x.flatten(), density=True, bins=bins
         ,alpha=0.5,label='data')
plt.plot(xs, pdfRadii(xs),label='analytical PDF') # plot N(x)=n(x)*4pi*x^2
plt.title("Just generated all_x")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('./test1.png')
#plt.show()
plt.close()
##### This part is for testing


print (f"a,b,c = {a,b,c}")


xs = np.linspace(1e-4,5,100)
bins = np.logspace(np.log10(1e-4),np.log10(5),21)
nperbin, _, _ = plt.hist(all_all_x.flatten(), density=True, bins=bins
         ,alpha=0.5,label='data')
plt.plot(xs, pdfRadii(xs),label='analytical PDF')
plt.title("All x and analytical pdf")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('./test2.png')
#plt.show()
plt.close()


# Linear interpolation is probably pretty good, as 
# we have a quite regularly sampled the grid
class LinearInterp3D(object):
    """
    Class for linear interpolation in 3D using regular grid input data
    Arguments:
    points -- the regular spaced grid [x_values,y_values,z_values]
              assumes these points are given in ascending order
              and at regular intervals of 0.1
              
    values -- the values corresponding to the regular spaced grid
    
    __call__ -- call with coordinate points (x,y,z) to calculate linear interpolated value
    
    """
    def __init__(self, points, values):
        self.points = points
        self.values = values
        
    def __call__(self, coordinates):
        # Interpolate at single coordinates (x,y,z), assumes these are not on the regular grid
        # Formulas from https://en.wikipedia.org/wiki/Trilinear_interpolation
        
        # Find the cube that bounds the x,y,z value
        indices = self.find_indices(coordinates)
        # Calculate differences
        xd = (coordinates[0]-self.points[0][indices[0]])/(self.points[0][indices[0]+1] - self.points[0][indices[0]])
        yd = (coordinates[1]-self.points[1][indices[1]])/(self.points[1][indices[1]+1] - self.points[1][indices[1]])
        zd = (coordinates[2]-self.points[2][indices[2]])/(self.points[2][indices[2]+1] - self.points[2][indices[2]])
        # Interpolate along x
        c00 = self.values[tuple(indices)]*(1-xd)+self.values[tuple(indices+np.array([1,0,0]))]*xd
        c01 = self.values[tuple(indices+np.array([0,0,1]))]*(1-xd) + self.values[tuple(indices+np.array([1,0,1]))]*xd
        c10 = self.values[tuple(indices+np.array([0,1,0]))]*(1-xd) + self.values[tuple(indices+np.array([1,1,0]))]*xd
        c11 = self.values[tuple(indices+np.array([0,1,1]))]*(1-xd) + self.values[tuple(indices+np.array([1,1,1]))]*xd
        # Interpolate along y
        c0 = c00*(1-yd) + c10*yd
        c1 = c01*(1-yd) + c11*yd
        # Interpolate along z
        c = c0*(1-zd) + c1*zd
        return c
        
    def find_indices(self, coordinates):
        """
        Find 3 indices in array points that define the lower bound
        of the box that bounds the coordinates
        We use the fact that we know the points are spaced at regular
        intervals of length 0.1
        """
        indices = []
        for i in range(3):
            indices.append( int( (coordinates[i]-self.points[i][0])/0.1 ) )
        return np.asarray(indices)
        
a_range = np.linspace(1.1,2.5,int((2.5-1.1)*10)+1)
b_range = np.linspace(0.5,2,int((2-0.5)*10)+1)
c_range = np.linspace(1.5,4,int((4-1.5)*10)+1)

# 3D array to save the results, shape = (15,16,26)
results = np.empty((len(a_range),len(b_range),len(c_range)))
for a_indx in range(0,len(a_range)):
    for b_indx in range(0,len(b_range)):
        for c_indx in range(0,len(c_range)):
            integ, error = romberg(lambda x: numbdensprofile(
                x, a_range[a_indx], b_range[b_indx], c_range[c_indx]
                , Nsat=Nsat, spherical=True) , 0, 5,order=10)
            integ *= prefactor
            # Normalize such that the integral produces <Nsat>
            A = Nsat/integ
            
            results[a_indx,b_indx,c_indx] = A

linInterp = LinearInterp3D([a_range,b_range,c_range],results)

# Thus we have to minimize the negative log-likelihood. 

# Better: Just calculate the sum of log(x_i) once
# then give that to this function since that will not change
def neglogL(a,b,c,xi):
    """
    Returns the negative log likelihood for a set of N i.i.d. 
    realizations of x_i 
    """
    N = len(xi)
    
    if (not(1.1 < a <2.5) or not(0.5 < b < 2) or not(1.5 < c < 4)):
        # a b or c is not in required bound, return high number
        return 1e6
    
    # We use the trilinear interpolator to approximate A(a,b,c)
    A = linInterp([a,b,c])
    
    # Dont know if we are allowed np.sum
    second_term = np.sum(np.log(xi))    
    second_term -= N*np.log(b)
    
    # Dont know if we are allowed to use np.sum
    return -1* ( N*np.log(A) + (a-1)*second_term - b**(-c) * np.sum(xi**c) )
    
def neglogL_scipy(x,xi):
    """
    Returns the negative log likelihood for a set of N i.i.d. 
    realizations of x_i 
    """
    N = len(xi)
    a,b,c = x
    
    if (not(1.1 < a <2.5) or not(0.5 < b < 2) or not(1.5 < c < 4)):
        # a b or c is not in required bound, return high number
        return 1e6
    
    # We use the trilinear interpolator to approximate A(a,b,c)
    A = linInterp([a,b,c])
    
    # Dont know if we are allowed np.sum
    second_term = np.sum(np.log(xi))    
    second_term -= N*np.log(b)
    
    # Dont know if we are allowed to use np.sum
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

def downhill_simplexND(func,start,delta,tol=2e-10,maxIter=20):
    """
    Find minimum of N-D function.
    
    func -- function to minimize (has to take x,y,.., as arguments)
    start -- vector of length N with the initial starting point
    delta -- guess for the characteristic length scale of the problem
    
    Returns
    x0 -- best guess for the minimum of the function
    vertices -- all vertices around the minimum
    it -- number of iterations done
    """
    N = len(start)
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
        order = np.argsort([func(*vertice) for vertice in vertices])
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
            vertices[-1] = x_try # Set new x_{N}
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
                vertices[-1] = x_try # I think this should be vertices[-1]
            else:
                # All options were apparently bad, just zoom in on best
                for i in range(1,N+1):
                    vertices[i] = 0.5*(centroid + vertices[i])

    return vertices[0], vertices, it
    
    
def read_in_halos(filename):
    """
    Read in the data for exercise 3, per filename.
    
    First number in the file on line 4 is always the number of halos.
    Then we find hash symbols to indicate a new halo,
    and then we find coordinates of satellites (if any)
    x, phi and theta.
    
    Returns the number of haloes in the datafile
    and the satellite positions divided per halo.
    """
    
    # List of lists of satellite positions x
    # every halo has a list of satellite positions x
    all_halos_satellites = []
    with open(filename, 'r') as file:
        for i, line in enumerate(file):
            if i == 3:
                num_halos = int(line.strip('\n'))
                halo_satellites = []
            
            # i = 4 is always the first halo.
            if i > 4:
                if '#' in line: # Next halo 
                    all_halos_satellites.append(halo_satellites)
                    halo_satellites = []
                else: # append satellite positions
                    coordinates = line.strip('\n').split('  ')
                    # Save only x position, cast to float
                    coordinates = float(coordinates[0])
                    halo_satellites.append(coordinates)
    
    # Don't forget to append the last satellites
    all_halos_satellites.append(halo_satellites)
    
    
    return num_halos, all_halos_satellites
            
            
# for i in [11,12,13,14,15]:
# Start with 14 for now
i=14
num_halos, all_halos_satellites = read_in_halos(f'./satgals_m{i}.txt')


import scipy.optimize as optimize

print (f"a,b,c = {a,b,c}")


xs = np.linspace(1e-4,5,100)
bins = np.logspace(np.log10(1e-4),np.log10(5),21)
nperbin, _, _ = plt.hist(all_all_x.flatten(), density=True, bins=bins
         ,alpha=0.5,label='data')
plt.plot(xs, pdfRadii(xs),label='analytical PDF')
plt.title("All x and analytical pdf")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('./test3.png')
#plt.show()
plt.close()

xs = np.linspace(1e-4,5,100)
bins = np.linspace((1e-4),(5),100)
nperbin, _, _ = plt.hist(all_all_x.flatten(), density=True, bins=bins
         ,alpha=0.5,label='data')
plt.plot(xs, pdfRadii(xs),label='analytical PDF')
plt.title("All x and analytical pdf")
#plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.savefig('./test3_1.png')
#plt.show()
plt.close()


# Function to minimize
minfunc = lambda x, y, z: neglogL(x,y,z,xi=all_all_x.flatten())
# Scipy adapted
minfunc_scipy = lambda x: neglogL_scipy(x,xi=all_all_x.flatten())

starting_point = [a+0.1,b+0.1,c+0.1]


print ("Scipy answer:")
opt = optimize.minimize(minfunc_scipy,starting_point,bounds=[(1.1,2.5),(0.5,2),(1.5,4)])
best_guess_scipy = opt.x
print (best_guess_scipy)

print ("Simplex:")
# Best guess simplex
best_guess, vertices, it = (downhill_simplexND(minfunc
                        ,starting_point,0.001
                          ,tol=2e-10,maxIter=2000))

print (best_guess, it)

# Plot scipy function
a,b,c = best_guess_scipy
A = linInterp(best_guess_scipy)

xs = np.linspace(1e-4,5,100)
bins = np.logspace(np.log10(1e-4),np.log10(5),21)
nperbin, _, _ = plt.hist(all_all_x.flatten(), density=True, bins=bins
         ,alpha=0.5,label='data')
plt.plot(xs, pdfRadii(xs),label='analytical PDF')
plt.title("Scipy params")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('./test4.png')
#plt.show()
plt.close()

xs = np.linspace(1e-4,5,100)
bins = np.linspace((1e-4),(5),100)
nperbin, _, _ = plt.hist(all_all_x.flatten(), density=True, bins=bins
         ,alpha=0.5,label='data')
plt.plot(xs, pdfRadii(xs),label='analytical PDF')
plt.title("Scipy params")
#plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.savefig('./test4_1.png')
#plt.show()
plt.close()

# Plot simplex function
a,b,c = best_guess
A = linInterp(best_guess)

xs = np.linspace(1e-4,5,100)
bins = np.logspace(np.log10(1e-4),np.log10(5),21)
nperbin, _, _ = plt.hist(all_all_x.flatten(), density=True, bins=bins
         ,alpha=0.5,label='data')
plt.plot(xs, pdfRadii(xs),label='analytical PDF')
plt.title("Simplex params")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('./test5.png')
#plt.show()
plt.close()

xs = np.linspace(1e-4,5,100)
bins = np.linspace((1e-4),(5),100)
nperbin, _, _ = plt.hist(all_all_x.flatten(), density=True, bins=bins
         ,alpha=0.5,label='data')
plt.plot(xs, pdfRadii(xs),label='analytical PDF')
plt.title("Simplex params")
#plt.xscale('log')
#plt.yscale('log')
plt.legend()
plt.savefig('./test5_1.png')
#plt.show()
plt.close()














                            

