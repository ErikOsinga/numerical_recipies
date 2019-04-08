import numpy as np

"""
This program contains routines of which I am not sure we can use a library
function for. Thus, I coded them up below
I assume very elementary math such as summing (np.sum) and 
raising to a power (np.pow) can just be used. 
"""


def linspace(lower,upper,numpoints):
    """
    Generate 'numpoints' equally spaced between 'lower' and 'upper'
    i.e., replacement of np.linspace
    """
    
    dx = (upper-lower)/(numpoints-1)
    point = lower
    ans = [point]
    for i in range(1,numpoints-1):
        point += dx
        ans.append(point)
    ans.append(upper)
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