import numpy as np
import matplotlib.pyplot as plt

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

# Randomly generate a,b,c within asked bounds
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

So in log log space, for x<1 we have an approx linear function 
log(n) \approx K * log(x)

And for x>1 we have an exponential function, so we fit this
in log-lin space with Neville's algorithm.
In this space we have approximately
log(n) \approx K * x^c

So, since c will be somewhere between 1.5 and 4, a quadratic
polynomial is a best guess. Thus we fit a polynomial of order 2
between last 3 datapoints with Nevilles algorithm
"""
# Linearly interpolate between first 3 datapoints
function = lambda x: densprofile(x,a,b,c,A=A,Nsat=Nsat
                                     ,spherical=False)
logfunc = lambda x: np.log(function(x))

points = np.array([1e-4, 1e-2, 1e-1, 1, 5])
ypoints = function(points)

# Linear interpolation, 100 datapoints between first 3 datapoints
x_interp_lin, y_interp_lin = linear_interpolation(points[:3],logfunc
                            , 100, 1e-4, 1e-1, logx=True)

# Nevilles method for 100 datapoints between last 3 datapoints
# x in linear space, y in logspace
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
plt.title('Combined interpolation in different spaces')
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
	Calculate the derivative of using Ridders method

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

