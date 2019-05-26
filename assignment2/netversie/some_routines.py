import numpy as np

"""
This program contains routines of which I am not sure we can use a library
function for. Thus, I coded them up below
I assume very elementary math such as summing (np.sum) and 
raising to a power (np.pow) can just be used. 

We also put the romberg integrator we wrote for assignment 1 here 
since a few programs need it.
"""


def linspace(lower,upper,numpoints):
    """
    Generate 'numpoints' equally spaced between 'lower' and 'upper'
    i.e., replacement of np.linspace
    """
    
    dx = (upper-lower)/(numpoints-1)
    ans = np.zeros(numpoints)
    ans[0] = lower
    for i in range(1,numpoints-1):
        ans[i] = ans[i-1] + dx
    ans[-1] = upper
    return np.asarray(ans)

def logspace(lower,upper,numpoints):
    """
    Generate 'numpoints' equally spaced on a log scale.
    i.e., replacement of np.logspace
    
    In linear space sequence starts at 10**(start) and ends at 10**(end)
    This function simply raises 10 to the power linspace
    """
    return np.power(10,linspace(lower,upper,numpoints))

def findmin(array):
    """
    Find minimum of array-like object 'array'
    """
    minimum = array[0]
    for i in range(1,len(array)):
        if array[i] < minimum:
            minimum = array[i]
            
    return minimum

def findmax(array):
    """
    Find maximum of array-like object 'array'
    """
    maximum = array[0]
    for i in range(1,len(array)):
        if array[i] > maximum:
            maximum = array[i]
            
    return maximum

def romberg(func, lbound, ubound, order=6, relacc=1e-14):
    """
    Calculate the integral of a function using Romberg's method
    with equal spaced abscissae
    
    func   -- function which gives the y values
    lbound -- lower bound of integral
    ubound -- upper bound of integral
    order  -- Amount of steps combining trapezoids, 
              final step will have 2**order intervals
    relacc -- target relative accuracy
    
    Returns 
    Value of the integral
    Error estimates at every step
     
    The error estimate is given as the difference between last 2 orders
    """
    
    # for saving the relative error
    relerror = [] # one value per column of the Romberg table
    # for saving S_i,j's
    all_S = np.zeros((order,order))
    
    i = 0
    delta_x = (ubound-lbound)
    points = linspace(lbound,ubound,2**i+1)
    integral = delta_x/2 * np.sum(func(points))
    all_S[0,0] = integral
    
    # Calculate the first column (S_{i,0})
    for i in range(1,order):
        delta_x /= 2
        # add points in the middle
        points = linspace(lbound,ubound,2**i+1)
        # add new points to the integral (use slicing to alternate)
        integral = 0.5*integral + delta_x * np.sum(func(points[1::2]))
        
        all_S[i,0] = integral
    
    # Calculate all others by combining
    for j in range(1,order): # column of Romberg table
        for i in range(j,order): # row of Romberg table
            all_S[i,j] = (4**j*all_S[i,j-1] - all_S[i-1,j-1]) / (
                           4**j - 1)
        # Relative error estimate is difference between last 
        # two estimates divided by the estimate 
        relerror.append(np.abs(1 - all_S[i,j-1]/all_S[i,j]) )
        if relerror[-1] < relacc:
#             print (f"Target relative error of {relacc} reached at S_{i,j}")
            # Target error reached, can stop
            return all_S[i,j], relerror
        
        if len(relerror) > 2:
            if relerror[-1] > relerror[-2]:
#                 print (f"Error increased at at S_{i,j}")
                # error increased, should stop
                return all_S[i,j],relerror        
    
    return all_S[order-1,order-1], relerror
