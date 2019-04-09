import numpy as np
import matplotlib.pyplot as plt

from some_routines import linspace, logspace, findmin, findmax

import question1 as q1

seed = q1.seed # Set seed only once 

def densprofile(x, a, b, c, A=1, Nsat = 100, spherical=False):
    """
    Returns the density profile from the assignment
    
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
    order -- Amount of steps combining trapezoids, 
    		 final step will have 2**order intervals
    
    Returns 
    Value of the integral
    Error estimate
     
    The error estimate is given as the difference between last 2 orders
    """
    
    # for saving S_i,j's
    all_S = np.zeros((order,order))
    
    i = 0
    delta_x = (ubound-lbound)
    points = linspace(lbound,ubound,2**i+1)
    integral = delta_x/2 * np.sum(func(points))
    all_S[0,0] = integral
    
    # Then calculate the first column (S_{i,0})
    for i in range(1,order):
        delta_x /= 2
        # add points in the middle
        points = linspace(lbound,ubound,2**i+1)
        # add new points to the integral (use slicing to alternate)
        integral = 0.5*integral + delta_x * np.sum(func(points[1::2]))
        
        all_S[i,0] = integral
    
    for j in range(1,order): # column of Romberg table
        for i in range(j,order): # row of Romberg table
            all_S[i,j] = (4**j*all_S[i,j-1] - all_S[i-1,j-1]) / (
                           4**j - 1)
    
    return all_S[order-1,order-1], (all_S[order-1,order-1]-all_S[order-1,order-2])

if __name__ == "__main__":
    # Randomly generate a,b,c within asked bounds
    RNGESUS = q1.RandomGenerator(seed=seed)

    a = RNGESUS.get_randomnumber()*(2.5-1.1) + 1.1
    b = RNGESUS.get_randomnumber()*(2.0-0.5) + 0.5
    c = RNGESUS.get_randomnumber()*(4-1.5) + 1.5

    # integral is only a function of x (or r) so add the prefactor manually
    prefactor = 4*np.pi # integral over theta and phi
    Nsat = 100
    print ("For the following ")
    print (f'a, b, c = {a,b,c}')
    integ, error = romberg(lambda x: densprofile(x, a, b, c, Nsat=Nsat
                            , spherical=True) , 0, 5,order=10)
    integ *= prefactor
    # Normalize such that the integral produces <Nsat>
    A = Nsat/integ
    print (f"We find A = {A}")

    # Make loglog plot to plot the points asked
    points = np.array([1e-4, 1e-2, 1e-1, 1, 5])
    ypoints = densprofile(points,a,b,c,A=A,Nsat=Nsat,spherical=False)
    plt.scatter(points,ypoints)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-4,5)
    plt.ylim(ypoints[-1],ypoints[0])
    plt.savefig('./plots/q2b1.png')
    plt.close()

def linear_interpolation(x, f, num_values, begin, end, logx=False):
    """
    Linear interpolation. Extrapolates as well if 'end' > 'x'

    Given 'x' values and function 'f'. Interpolates linearly 
    'num_values' between 'begin' and end'
    if logx = True, use equally spaced x values in logspace
    otherwise, use equally spaced x in linear space
    
    The function always takes all abcissae in linear space.
    Whether y is logspace is determined by the function 'f' 

    Returns 
    x_values -- (log) abcissae values where function was interpolated
    y_values -- interpolated value
    
    """
    # y values of the points that are given
    y = f(x)
    # interpolated y values
    y_values = []
    # x values to interpolate at
    if logx:
        # equal width in log space
        x_values = logspace(np.log10(begin),np.log10(end),num_values)
        # interpolate in log(x) space
        x_values = np.log(x_values)
        x = np.log(x)
    else:
        # interpolate in linear space
        x_values = linspace(begin,end,num_values)
    
    # Interpolation
    for i in range(len(x)-1):
        # calculate slope between two points
        a = ( y[i+1] - y[i] ) / (x[i+1] - x[i])
        # take only x values between two points
        x_values_now = x_values[(x_values >= x[i]) & (x_values < x[i+1])]
        # calculate y values as linear interpolation
        y_values += list(y[i] + (x_values_now - x[i])*a)
        
    # Extrapolation, if needed, is simply extrapolating final bin
    x_values_now = x_values[(x_values >= x[i+1])]
    y_values += list(y[i] + (x_values_now - x[i])*a)
    
    return x_values, np.asarray(y_values)

def recurrence_relation(i, j, x, all_x, f,logx=False):
    """
    Neville's algorithm recurrence relation. 
    Fits a polynomial of order N-1 through N points

    i,j -- int   -- index of the datapoints 
    x   -- float -- x value to evaluate
    all_x -- array -- data points x
    f   -- function -- function that calculates y(x)
    """
    if i == j:
        # return y_i
        if logx:
            return f(np.exp(all_x[i]))
        else:
            return f(all_x[i])
    else:
        # return the recursive relation
        return ( ((x - all_x[j])*recurrence_relation(i, j-1, x, all_x, f, logx) 
         - (x-all_x[i])*recurrence_relation(i+1, j, x, all_x, f,logx)) /
        (all_x[i] - all_x[j]) ) 

# Combine linear with Neville's for the final result
"""
# Log-log space is the preferred space for the start of the function
because if we take the logarithm
of both sides, the expression becomes 
log(n) \propto log(x)- x^c

So in log log space, for x < \approx 1 we have an approx linear function 
log(n) \approx K * log(x)

And for x>1 we have an exponential function, so we fit this
in log-lin space with Neville's algorithm.
In this space we have approximately
log(n) \approx K * x^c

So, since c will be somewhere between 1.5 and 4, a quadratic
polynomial is a best guess. Thus we fit a polynomial of order 2
between last 3 datapoints with Nevilles algorithm
"""
if __name__ == "__main__":
    # Linearly interpolate between first 3 datapoints, in log-log space
    function = lambda x: densprofile(x,a,b,c,A=A,Nsat=Nsat
                                         ,spherical=False)
    logfunc = lambda x: np.log(function(x))

    points = np.array([1e-4, 1e-2, 1e-1, 1, 5])
    ypoints = function(points)

    # Linear interpolation, 100 datapoints between first 3 datapoints in log-log space
    x_interp_lin, y_interp_lin = linear_interpolation(points[:3],logfunc
                                , 100, 1e-4, 1e-1, logx=True)

    # Nevilles method for 100 datapoints between last 3 datapoints
    # x in linear space, y in logspace. 
    x_interp_pol = logspace(np.log10(1e-1), np.log10(5), 100)
    y_interp_pol = [recurrence_relation(0,len(points)-3, x_interp_pol[i], 
                points[2:], logfunc, logx=False) 
                for i in range(len(x_interp_pol)) ]

    # Combine them and transform to first x half to lin space
    # and both y halves to lin space for plotting
    x_interp = np.append(np.exp(x_interp_lin),x_interp_pol)
    y_interp = np.exp(np.append(y_interp_lin,y_interp_pol))

    plt.plot(x_interp,y_interp,label='Interpolation',c='g')
    # plt.plot(x_interp, function(x_interp),label='True function', ls='dashed')
    plt.scatter(points,ypoints, c='k',label='Datapoints')
    plt.legend()
    plt.title('Two interpolation methods in different spaces')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('./plots/q2b2.png')
    plt.close()



# Numerically calculate dn(x)/dx at x=b
def anal_deriv(x,a,b,c,A,Nsat):
    """Analytical derivative of density profile"""
    return ( (a-3-c*(x/b)**c) * (x**(a-4))* np.exp( -(x/b)**c ) 
        *A*Nsat/(b**(a-3)) )

def central_difference(func, x, h):
    """
    calculate numerical derivative with central difference method
    
    """
    return (func(x+h) - func(x-h)) / (2*h)

def ridders_method(func, x, d, m, target_error):
    """
	Calculate the derivative using Ridders method

    func -- function to calculate numerical derivative for
    x -- position at which it is calculated
    d -- factor with which h of the central difference is reduced every step
    m -- highest order before function is terminated
    target_error -- target error, once it is reached, function stops

    Returns
    best value for derivative
    approximation of the error
    """
    
    # for saving the error, one value per column
    error = []
    # for saving D_i,j's (Ridders table)
    all_D = np.zeros((m,m))
    
    # First approximation
    h = 0.1
    all_D[0,0] = central_difference(func, x, h)
    
    # Calculate the first column (D{i,0})
    for i in range(m):
        h /= d
        all_D[i,0] = central_difference(func, x, h)
    
    # Calculate all others by combining
    for j in range(1,m): # columns
        for i in range(j,m): # rows
            all_D[i,j] = (d**(2*(j+1)) * all_D[i,j-1] - all_D[i-1,j-1]) / (
                           d**(2*(j+1)) - 1)
        error.append(np.abs(all_D[i,j]-all_D[i,j-1]))
        if error[-1] < target_error:
#             print (f"Target error of {target_error} reached at D_{i,j}")
            return all_D[i,j], error
        if len(error) > 2:
            if error[-1] > error[-2]:
#                print (f"Error increased at at D_{i,j}")
                # error increased, should stop
                return all_D[i,j], error

    return all_D[m-1,m-1], error

if __name__ == "__main__":
    # Output the numerical derivative alongside the analytical one 
    # to at least 12 significant digits.
    print (f"dn(x)/dx at x=b={b}")
    ridder, error = ridders_method(function, b, d=2, m=15, target_error=1e-15)
    print(f'Numerical derivative: {ridder:.12f}')

    analderiv = anal_deriv(b,a,b,c,A,Nsat)
    print(f'Analytical derivative: {analderiv:.12f}')


def pdfRadii(x, a, b, c, A):
    """
    PDF for question 2d
    """
    # densprofile (*x^2) with Nsat = 1 so we effectively divide by Nsat
    return ( densprofile(x,a,b,c,A,Nsat=1,spherical=True) * 4*np.pi)

def rejection_sampling(RNG, pdf, x_begin, x_end, num_points=500):
    """
    Sample num_points from the PDF by rejection sampling
    
    RNG -- random number generator that generates random float \in [0,1]
    pdf -- pdf of the function to sample from
    num_points -- amount of points to sample
    x_begin, x_end -- begin and endpoint of uniform dist sample

    Returns

    all_x -- list of sampled values
    """
        
    all_x = []
    factor = x_end-x_begin
    while len(all_x) < num_points:
        x = RNG()*(factor)+x_begin 
        y = RNG()
        if y <= pdf(x):
            all_x.append(x)
    
    return all_x

if __name__ == "__main__":
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

def output_N_satellites(RNG, pdf, N):
    """
    Generate N satellites following the density profile
    given as 'pdf'
    
    We assume 
    0<=x<=5, 0<=\phi<2\pi, 0<= theta <= \pi
    
    Inputs
    RNG -- a function that will generate x ~ unif[0,1] when called
    pdf -- the PDF to generate the data from
    N -- number of satellites

    Returns
    all_x -- list of N randomly generated x positions
    all_theta -- list of N randomly generated theta positions
    all_phi -- list of N randomly generated phi positions
    """
    
    # Generate x with rejection sampling,
    # because we cannot invert or integrate the PDF
    all_x = rejection_sampling(RNG, pdf, 0, 5, N) # x = r/rvir

    # Generate theta with inverse transform sampling
    # because integrating and inverting is very easy
    all_theta = inverse_transform_sample(RNG, invcdf, N)
    
    # Generate phi with just uniform numbers between 0 and 2pi
    all_phi = []
    for i in range(N):
        all_phi.append(RNG())
    all_phi = np.array(all_phi)*2*np.pi

    return all_x, all_theta, all_phi
    
if __name__ == "__main__":
    # Output the positions for 100 such satellites, with current a,b,c and A
    thispdf = lambda x: pdfRadii(x,a,b,c,A)
    all_x, all_theta, all_phi = output_N_satellites(RNG,thispdf,N=100)
    np.savetxt('./satellitepositions.txt',np.transpose([all_x,all_theta,all_phi]))
    # Positions are output in PDF file 

    # Repeat d) for 1000 halos with 100 satellites each.
    all_all_x, all_all_theta, all_all_phi = [], [], []
    for i in range(1000):
        all_x, all_theta, all_phi = output_N_satellites(RNG,thispdf, N=100)
        all_all_x.append(all_x)
        all_all_theta.append(all_theta)
        all_all_phi.append(all_phi)

    all_all_x = np.asarray(all_all_x)

    # Make another log-log plot, showing N(x).
    xs = linspace(1e-4,5,100) # sampled along x
    # Generate 20 logarithmically equal spaced bins between 1e-4 and 5
    bins = logspace(np.log10(1e-4),np.log10(5),21)
    # Plot it as histogram
    nperbin, _, _ = plt.hist(all_all_x.flatten(), bins=bins
                             ,alpha=0.5,label='data')
    plt.close() # We could also set density is true, but since I am unsure
    # whether this is allowed, we shall normalize it manually.

    bin_centers = (bins[:-1] + bins[1:])/2
    binwidths = (bins[1:] - bins[:-1])
    # Divide each bin by its width
    # And divide by the total count to normalize
    nperbin /= binwidths*np.sum(nperbin)

    # Normalized histogram
    plt.bar(bin_centers,nperbin,binwidths,label='data',alpha=0.5)
    # Analytical function
    plt.plot(xs, thispdf(xs),label='analytical PDF',c='C1') # plot N(x)=n(x)*4pi*x^2

    plt.legend(frameon=True)
    plt.xlabel('x')
    plt.ylabel('Probability or normalized counts')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(1e-7,findmax([findmax(thispdf(xs)),findmax(nperbin)]))
    plt.savefig('./plots/q2e1.png')
    plt.close()
    # The galaxies match the distribution very well


# Root finding 
def position_of_maxNx(a,b,c):
    """
    Analytical formula for position x of the maximum of 
    N(x) = n(x)*4pi*x^2, given a b and c
    """
    return b*( (a-1)/c )**(1/c)

def false_position(func, lower, upper, acc, MAX=100):
    """
    Fit secant lines through two points to approximate the root
    Update points such that the root remains bracketed
    
    func -- function to approximate the root of
    lower -- first abcissa point of initial guess 
    uppper -- second abcissa point of initial guess 
    acc -- target accuracy
    MAX -- maximum number of iterations before function stops
    
    Returns
    x1,x2 -- two points that bracket the root
    i+1   -- number of iterations done
    """
    
    ylower, yupper = func(lower), func(upper)
    
    if ylower * yupper >= 0:
        raise ValueError("Incorrect starting bracket")
    # smallest function value is 'most recent' guess = x2
    if abs(ylower) <= abs(yupper):
        x2 = lower
        x1 = upper
    else:
        x2 = upper
        x1 = lower
    
    for i in range(MAX):        
        # new value
        x3 = x1 - ( func(x1)*(x1-x2)  / (func(x1) - func(x2)) )
        # update values, most recent guess is two values that bracket
        if func(x3)*func(x2) <= 0: 
            # if new point has different sign, then update last point
            x1 = x2
        else:
            x1 = x1
        x2 = x3
        
        if abs(x1 - x2) < acc:
            break
        
    return x1, x2, i+1

if __name__ == "__main__":
    # Use analytical formula to find the position of the maximum
    xmax = position_of_maxNx(a,b,c)
    ymax = thispdf(xmax)

    # Then use root finding to find where the function-ymax/2 is 0
    function = lambda x: thispdf(x) - ymax/2

    # Secant is not a good option, because we will likely diverge 
    # Due to the linear approximation of the function not being very good
    # Therefore, we use the false position method, since it will not diverge
    # And we know very well where the roots will be

    acc = 1e-8
    # First root will be between starting value and x of the maximum
    xlower1, xupper1, itneeded1 = false_position(function, 1e-4,xmax,acc)
    # Second root will be x of the maximum and 5
    xlower2, xupper2, itneeded2 = false_position(function, xmax,5, acc)

    best_guess1 = (xlower1+xupper1)/2
    best_guess2 = (xlower2+xupper2)/2
    print (f"First root is approximately at {best_guess1}")
    # print (f"{itneeded1} iterations were needed")

    print (f"Second root is approximately at {best_guess2}")
    # print (f"{itneeded2} iterations were needed")


def find_position_maximum(array):
    """
    Given a 1D array-like object, finds the index of the max value
    
    returns imax -- the index of the maximum value in the array
    """
    maximum = array[0]
    imax = 0
    for i in range(1,len(array)):
        if array[i] > maximum:
            maximum = array[i]
            imax = i
            
    return imax

def quicksort(arr):
    """
    Sort array with quicksort. Recursively apply quicksort on sub-arrays.
    sorting is performed IN PLACE
    """
    N = len(arr)
    
    # Make sure first/last/middle elements are ordered correctly
    if arr[0] > arr[N-1]: # swap leftmost and rightmost elements
        arr[0], arr[N-1] = arr[N-1], arr[0]
    if arr[(N-1)//2] > arr[N-1]: # swap middle and rightmost element
        arr[(N-1)//2], arr[N-1] = arr[N-1], arr[(N-1)//2]
    if arr[0] > arr[(N-1)//2]: # swap middle and leftmost element
        arr[0], arr[(N-1)//2] = arr[(N-1)//2], arr[0]
    
    i, j = 0, N-1
    pivot = arr[(N-1)//2]
    pivot_position = (N-1)//2
    for _ in range(0,N//2): 
        while arr[i] < pivot:
            i +=  1
        while arr[j] > pivot:
            j -= 1
        if j <= i:
            break # pointers have crossed
        else:
            if i == pivot_position: # have to keep track of where the pivot is
                pivot_position = j # going to switch j with pivot
            elif j == pivot_position:
                pivot_position = i # going to switch i with pivot

            # Both i and j found, swap them around the pivot and continue
            arr[i], arr[j] = arr[j], arr[i]
    
    if N > 2:
        # As long as we don't have 1 element arrays, perform quicksort on the subarrays
        leftarr = arr[:pivot_position] # left of the pivot
        rightarr = arr[pivot_position+1:] # right of the pivot        
        quicksort(leftarr)
        quicksort(rightarr)

def give_percentile(sorted_arr, percentile):
    """
    Percentile defined with midpoint interpolation.

    sorted_arr -- sorted array to calculate percentile from
    percentile -- percentile to return
    """
    N = len(sorted_arr)
    index = percentile/100 * (N-1)
    if index == int(index):
        return sorted_arr[int(index)]
    else:
        # Midpoint interpolation
        index = int(index)
        return (sorted_arr[index] + sorted_arr[index+1])/2
        
# Linear interpolation is probably pretty good, as 
# we have a quite regularly sampled grid
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
        xd = (coordinates[0]-self.points[0][indices[0]])/(self.points[0][indices[0]+1]
                                                         - self.points[0][indices[0]])
        yd = (coordinates[1]-self.points[1][indices[1]])/(self.points[1][indices[1]+1]
                                                         - self.points[1][indices[1]])
        zd = (coordinates[2]-self.points[2][indices[2]])/(self.points[2][indices[2]+1]
                                                         - self.points[2][indices[2]])
        # Interpolate along x, casting indices to tuple to get correct elements
        c00 = self.values[tuple(indices)]*(1-xd)+self.values[tuple(indices+np.array([1,0,0]))]*xd
        c01 = self.values[tuple(indices+np.array([0,0,1]))]*(1-xd) + self.values[tuple(
                                                        indices+np.array([1,0,1]))]*xd
        c10 = self.values[tuple(indices+np.array([0,1,0]))]*(1-xd) + self.values[tuple(
                                                        indices+np.array([1,1,0]))]*xd
        c11 = self.values[tuple(indices+np.array([0,1,1]))]*(1-xd) + self.values[tuple(
                                                        indices+np.array([1,1,1]))]*xd
        # Interpolate along y
        c0 = c00*(1-yd) + c10*yd
        c1 = c01*(1-yd) + c11*yd
        # Interpolate along z
        c = c0*(1-zd) + c1*zd
        return c
        
    def find_indices(self, coordinates):
        """
        Find 3 indices in array points that define the lower bound
        of the box that bounds the coordinates.
        We use the fact that we know the points are spaced at regular
        intervals of length 0.1
        """
        indices = []
        for i in range(3):
            indices.append( int( (coordinates[i]-self.points[i][0])/0.1 ) )
        return np.asarray(indices)



if __name__ == "__main__":
    # Bin containing largest number of galaxies
    max_n = find_position_maximum(nperbin)
    # boundaries of the bin
    lower_x = bins[max_n]
    upper_x = bins[max_n+1]


    # Find the x values of the satellites per halo
    # that are in the specified bin. Shape (1000,?)
    halo_satellites_in_bin = []
    for i in range(0,all_all_x.shape[0]):
        satellites_in_bin = []
        for j in range(0,all_all_x.shape[1]):
            if all_all_x[i,j] > lower_x and all_all_x[i,j] < upper_x:
                satellites_in_bin.append(all_all_x[i,j])
        halo_satellites_in_bin.append(satellites_in_bin)
        
    # Concatenate (i.e., flatten) the list of lists
    total_satellites_in_bin = np.concatenate(halo_satellites_in_bin)
    # Sorted
    quicksort(total_satellites_in_bin)

    median = give_percentile(total_satellites_in_bin,50)
    sixteent = give_percentile(total_satellites_in_bin, 16)
    eightyfth = give_percentile(total_satellites_in_bin, 84)

    print ("Maximum number in bin", max_n, "which is between x's:")
    print (bins[max_n:max_n+2])

    print (f"Median x: {median}")
    print (f"16th PCTL: {sixteent}")
    print (f"84th PCTL: {eightyfth}")

    # Now make a histogram of the number of galaxies in this radial bin in each halo

    # Thus, in each halo, store the number of satellites in the specified bin
    num_satellites_bin = [] # shape (1000,)
    for halo in range(0,len(halo_satellites_in_bin)):
        # amount of satellites within the specified bin in this halo
        amount_in_bin = (len(halo_satellites_in_bin[halo]))
        num_satellites_bin.append(amount_in_bin)

    # Each bin should have a width of 1
    max_num = findmax(num_satellites_bin)
    min_num = findmin(num_satellites_bin)
    bins = linspace(min_num, max_num, max_num-min_num+1)

    mean_num = np.mean(num_satellites_bin)
    poisson_prob = q1.poisson_probability(bins,mean_num)

    nperbin, _, _ = plt.hist(num_satellites_bin, bins=bins
                             ,alpha=0.5,label='data')
    plt.close() # We could also set density is true, but since I am unsure
    # whether this is allowed, we shall normalize it manually.
    bin_centers = (bins[:-1] + bins[1:])/2
    binwidths = (bins[1:] - bins[:-1])
    # Divide each bin by its width
    # And divide by the total count to normalize
    nperbin /= binwidths*np.sum(nperbin)

    # Normalized histogram
    plt.bar(bin_centers,nperbin,binwidths,label='data',alpha=0.5)
    plt.plot(bins,poisson_prob,label='Poisson PDF',color='C1')
    plt.xlabel('Number of galaxies in specified bin')
    plt.ylabel('Normalized counts or probability')
    plt.legend()
    plt.savefig('./plots/q2g1.png')
    plt.close()


    # Interpolation

    # Normalization factor A depends on a,b,c. Calculate a regularly spaced A grid.
    a_range = linspace(1.1,2.5,int((2.5-1.1)*10)+1) # 0.1 wide intervals
    b_range = linspace(0.5,2,int((2-0.5)*10)+1)
    c_range = linspace(1.5,4,int((4-1.5)*10)+1)

    prefactor = 4*np.pi
    Nsat = 100

    # 3D array to save the results, shape = (15,16,26)
    results = np.empty((len(a_range),len(b_range),len(c_range)))
    for a_indx in range(0,len(a_range)):
        for b_indx in range(0,len(b_range)):
            for c_indx in range(0,len(c_range)):
                # Calculate the integral
                integ, error = romberg(lambda x: densprofile(
                    x, a_range[a_indx], b_range[b_indx], c_range[c_indx]
                    , Nsat=Nsat, spherical=True) , 0, 5,order=10)
                integ *= prefactor
                # Normalize such that the integral produces <Nsat>
                A = Nsat/integ
                
                results[a_indx,b_indx,c_indx] = A

    # Save results, because we need it in question 3
    np.save('./A_values_grid.npy',results)

    # Construct the interpolator
    linInterp = LinearInterp3D([a_range,b_range,c_range],results)
    # Example of how to interpolate a point:  linInterp([1.41,0.65,2.57]) 
