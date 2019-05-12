

Idee: Calculate alleen de additional area. ToDo later.


def GaussianCdf(x):
    """
    Calculate the Gaussian CDF at point(s) x
    
    If x is an array, we expect it to be sorted. Since if x is sorted
    we can simply add to the calculation of previous x 
    """
    
    # the error function is the integral of this erfarg
    # from 0 to x/sqrt(2), multiplied by 2/sqrt(pi)
    erfarg = lambda t: np.exp(-t**2)
    
    if type(x) == np.ndarray or type(x) == list:
        if type(x) == list: 
            x = np.array(x)
        # Calculate the CDF at the first value with high order Romberg
        xnow = x[0]
        cdf = 0.5*(1+2/np.sqrt(np.pi)*romberg(erfarg,0,xnow/np.sqrt(2)
                                              ,order=10)[0])
        all_cdf = [cdf]
        # for all other points, only calculate the additional area
        diff = np.diff(x)
        for xnow in diff:
            extra_area = 0.5*(1+2/np.sqrt(np.pi)*romberg(erfarg,0
                            ,xnow/np.sqrt(2),order=12)[0])
            
        
    else:
        cdf = 0.5*(1+2/np.sqrt(np.pi)*romberg(erfarg,0,x/np.sqrt(2)
                                              ,order=12)[0])
    return cdf
