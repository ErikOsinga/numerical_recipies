#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest
from astropy.stats import kuiper
import some_routines as sr


# # 1. Normally distributed pseudo-random numbers
# #### a) an RNG that returns a float in [0,1]

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
        # the seed for the MWC, has to be smaller than 2**32
        self.X3 = dtyp(seed)
        if self.X3 >= 2**32:
            raise ValueError("Please provide a seed smaller than 2**32")
        
        self.max_value = dtyp(2**64 - 1)
        
        # LCG values from Numerical Recipies
        self.a = dtyp(1664525)
        self.c = dtyp(1013904223)
        self.m = dtyp(2**32)
        
        # 64 bit XOR shift values from Numerical Recipes
        self.a1, self.a2, self.a3 = dtyp(21), dtyp(35), dtyp(4)
        
        # MWC values from Numerical Recipes
        self.a2 = dtyp(4294957665)
        self.maxx = dtyp(2**32-1)
        self.shift = dtyp(32)
        
    def lincongen(self, X):    
        return (self.a*X+self.c) % self.m

    def XORshift64(self, X):
        if X == 0:
            raise ValueError("Seed cannot be zero")
        X = X ^ (X >> self.a1)
        X = X ^ (X << self.a2)
        X = X ^ (X >> self.a3)
        
        return X
    
    def MWC(self, X):
        X = self.a2*(X & self.maxx) + (X >> self.shift)
        # Use as a random number only lowest 32 bits
        # But in a bitmix, we can use all 64 bits.
        return X
    
    def get_randomnumber(self):
        """
        Combine LCG and XORshift to produce random float 
        between 0 and 1
        """
        self.X1 = self.lincongen(self.X1)
        self.X2 = self.XORshift64(self.X2)
        self.X3 = self.MWC(self.X3)
        
        # output is XOR of these numbers
        
        return ((self.X1^self.X3)^self.X2)/self.max_value
    
RNGESUS = RandomGenerator(seed=seed)

# Generate 1000 random numbers. Plot comparison
all_randnum = np.zeros(1000)
for i in range(1000):
    all_randnum[i] = RNGESUS.get_randomnumber()

plt.plot(all_randnum,np.roll(all_randnum,1),'o',alpha=0.5)
plt.title(f'Comparison of {len(all_randnum)} random numbers')
plt.xlabel('Value element n')
plt.ylabel('Value element n+1')
plt.savefig('./plots/q1a1.png')
plt.close()

plt.scatter(range(0,len(all_randnum)),all_randnum,s=10)
# Connect them to see whether they are correlated
plt.plot(all_randnum,lw=1,ls='solid',alpha=0.5)
plt.title(f'First {len(all_randnum)} random numbers')
plt.ylabel('Value element')
plt.xlabel('Iteration')
plt.savefig('./plots/q1a2.png')
plt.close()

# Now generate 1 million random numbers. Plot histogram
all_randnum = np.zeros(int(1e6))
for i in range(int(1e6)):
    all_randnum[i] = RNGESUS.get_randomnumber()

plt.hist(all_randnum,bins=sr.linspace(0,1,21),edgecolor='black')
plt.title(f'Histogram of 1 million random numbers')
plt.xlabel('Value')
plt.ylabel('Counts')
plt.savefig('./plots/q1a3.png')
plt.close()


# #### b. Now use the Box-Muller method to generate 1000 normally distributed 
# random numbers.

def BoxMuller(randnums):
    """
    Given an input of an even number of 
    random numbers drawn from Unif(0,1)
    Return the same amount of random numbers drawn from Gaussian(0,1)
    """
    z1, z2 = randnums[:len(randnums)//2], randnums[len(randnums)//2:]
    a = np.sqrt(-2*np.log(z1))
    x1 = a*np.cos(2*np.pi*z2)
    x2 = a*np.sin(2*np.pi*z2)
    
    randnums2 = np.concatenate([x1,x2])
    
    return randnums2

def GaussianTransform(x,mu,sigma):
    """
    Takes x's drawn from a standard normal distribution and maps them 
    to arbitrary x ~ G(mu,sigma)
    """
    x *= sigma
    x += mu
    return x

def GaussianPdf(x,mean,sigma):
    """return the PDF of a Gaussian"""
    variance = sigma**2
    return 1/(np.sqrt(2*np.pi*variance)) * np.exp(-0.5*(x-mean)**2/variance)


# We can use the first 1000 random numbers we have generated
randgauss = BoxMuller(all_randnum[:1000])
mu, sigma = 3, 2.4
randgauss = GaussianTransform(randgauss,mu,sigma)
# 20 equal width bins
hbins = sr.linspace(mu-3*sigma, mu+3*sigma,21)
nperbin, _, _ = plt.hist(randgauss,bins=hbins,label='data')
plt.close() # We could also set density is true, but since I am unsure
    # whether this is allowed, we shall normalize it manually.
bin_centers = (hbins[:-1] + hbins[1:])/2
binwidths = (hbins[1:] - hbins[:-1])
# Divide each bin by its width
# And divide by the total count to normalize
nperbin /= binwidths*np.sum(nperbin)

# Normalized histogram
plt.bar(bin_centers,nperbin,binwidths,label='Data',alpha=0.5)
# Analytical pdf
xs = sr.linspace(mu-5*sigma,mu+5*sigma,101)
plt.plot(xs, GaussianPdf(xs,mu,sigma),label='Analytical PDF',c='C1')

# Indicate theoretical 1 to 4 sigma intervals with a line
for i in range(1,5):
    plt.axvline(mu-i*sigma,ls='dashed',alpha=0.5,c='k')
    plt.text(mu-i*sigma,0.1,f'{i}$\sigma$',rotation=90)
    plt.axvline(mu+i*sigma,ls='dashed',alpha=0.5,c='k')

plt.legend(frameon=False)
plt.xlabel('x')
plt.ylabel('Probability or normalized counts')
plt.xlim(mu-5*sigma,mu+5*sigma)
plt.savefig('./plots/q1b1.png')
plt.close()


# #### c. Write a code that can do the KS-test on your function to determine 
# whether it is consistent with a normal dist. For this, use $\mu=0$ and $\sigma=1$.
# Make a plot of the probability that your Gaussian random number generator is 
# consistent with Gaussian distributed random numbers. 
# Start with 10 random numbers and use in the plot a spacing of 0.1 dex until 
# you have calculated it for $10^5$ random numbers on the x-axis. 
# Compare your algorithm with the KS test from scipy by making another plot 
# with the result from your KS-test and the KS-test from scipy.

def GaussianCdf(x):
    """
    Calculate the Gaussian CDF at point(s) x    
    """
    
    # the error function is the integral of this erfarg
    # from 0 to x/sqrt(2), multiplied by 2/sqrt(pi)
    erfarg = lambda t: np.exp(-t**2)
    
    if type(x) == np.ndarray or type(x) == list:
        if type(x) == list: 
            x = np.array(x)
        cdf = np.zeros(len(x))
        for i, xnow in enumerate(x):
            cdfn = 0.5*(1+2/np.sqrt(np.pi)*sr.romberg(erfarg,0,xnow/np.sqrt(2)
                                              ,order=6)[0])
            cdf[i] = cdfn            
    else:
        cdf = 0.5*(1+2/np.sqrt(np.pi)*sr.romberg(erfarg,0,x/np.sqrt(2)
                                              ,order=6)[0])
    return cdf # returns a float or numpy array

def quicksort(arr):
    """
    Sort array with quicksort
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
    for _ in range(0,N//2): # Can improve this range()
        while arr[i] < pivot:
            i +=  1
        while arr[j] > pivot:
            j -= 1
        if j <= i:
            break # pointers have crossed
        else:
            if i == pivot_position: # have to keep track of where the pivot is
                pivot_position = j # going to switch them
            elif j == pivot_position:
                pivot_position = i
            # Both i and j found, swap them around the pivot and continue
            arr[i], arr[j] = arr[j], arr[i]
    
    if N > 2:
        # As long as we don't have 1 element arrays, perform quicksort on the subarrays
        leftarr = arr[:pivot_position] # left of the pivot
        rightarr = arr[pivot_position+1:] # right of the pivot        
        quicksort(leftarr)
        quicksort(rightarr)

def KScdf(z):
    """
    Return the CDF of the KS distribution at 'z'
    Used by the function KStest where 'z' is a function of 
    the KS test statistic 'D'
    """
    pi = np.pi
    exp = np.exp
    
    if z < 1.18:
        term = exp(-pi**2/(8*z**2))
        ans = np.sqrt(2*pi)/z 
        ans *= (term + term**9 + term**25)
        
    else:
        term = exp(-2*z**2)
        ans = 1-2*(term-term**4+term**9)
        
    return ans
        
def KStest(x, hCDF, sorted=False):
    """
    Given an array of observations of x drawn from some PDF,
    this function uses the (two-sided) KS test to test whether x 
    follows the null hypothesis CDF
    
    x      -- array:    realisation from some PDF 
    hCDF   -- array:    CDF of null hypothesis evaluated at 'x'
    sorted -- boolean:  whether the data is already sorted
    
    Returns
    D    -- float: KS test statistic
    pval -- float: p-value to reject null hypothesis 
        
    """
    N = len(x)
    
    if not sorted: # Sort x in ascending order to approximate the ECDF
        # Find the index that sorts the current points
        indices = quickargsort(x)
        # Sort the current points
        x = x[indices]    
        # Make sure to swap the null hypothesis CDF values equivalently
        hcDF = hCDF[indices]

    # empirical cdf
    ECDF = sr.linspace(1,N,N)/N
    # Maximum distance above
    Dplus = sr.findmax(ECDF-hCDF)
    # Maximum distance below, use the left points of bins
    Dmin = sr.findmax(hCDF - sr.linspace(0,N-1,N)/N)
    
    # KS test statistic D, in a two-sided test, is the maximum of these
    D = sr.findmax([Dmin,Dplus])
    
    sqN = np.sqrt(N)
    z = (sqN+0.12+0.11/sqN)*D
    pval = 1-KScdf(z)
    
    return D, pval

def Kuiperpvalue(lamb,acc=1e-8,maxIT=100):
    """
    Returns the p value of the Kuiper test given the right lamb
    See the function KuiperTest for the definition of lamb 
    in terms of the Kuiper statistic V.
    
    Uses a series for the asymptotic distribution of the statistic V
    See Numerical Recipes (3rd Edition) Equation (14.3.23)
    """
    if lamb < 0.4:
        return 1.0 # accurate to 7 digits
    else:
        diff = acc+1
        j = 0
        Q = 0
        while diff > acc:
            j += 1
            Qold = Q
            Q += (4*j**2*lamb**2-1)*np.exp(-2*j**2*lamb**2)
            diff = np.abs(Q-Qold)

            
            if j > maxIT:
                print (f"Not converged to the required accuracy of {acc}")
                break           
    return 2*Q
    
def KuiperTest(x, hCDF, sorted=False):
    """
    Given an array of observations of x drawn from some PDF,
    this function uses the KS test to test whether x 
    follows the null hypothesis CDF
    
    x      -- array:    realisation from some PDF 
    hCDF   -- array:    CDF of null hypothesis evaluated at 'x'
    sorted -- boolean:  whether the data is already sorted
    
    Returns
    V    -- float: Kuiper test statistic
    pval -- float: p-value to reject null hypothesis 
        
    """
    N = len(x)
    
    if not sorted: # Sort x in ascending order to approximate the ECDF
        # Find the index that sorts the current points
        indices = quickargsort(x)
        # Sort the current points
        x = x[indices]    
        # Make sure to swap the null hypothesis CDF values equivalently
        hcDF = hCDF[indices]
    
    # empirical cdf
    ECDF = sr.linspace(1,N,N)/N
    # Maximum distance above
    distance = ECDF-hCDF
    Dplus = sr.findmax(distance) 
    # Maximum distance below, use the left points of bins
    Dminus = sr.findmax(hCDF - sr.linspace(0,N-1,N)/N)
    V = Dplus + Dminus # Kuiper Test statistic
    
    sqN = np.sqrt(N)
    lamb = (sqN+0.155+0.24/sqN)*V
    pval = Kuiperpvalue(lamb)
    
    return V, pval

def KStest_2sample(x1, x2, sorted=False):
    """
    Compute the 2 sample KS Test on two datasets x1 and x2
    
    x1     -- array:    realisation from some PDF 
    x2     -- array:    realisation from some PDF
    sorted -- boolean:  whether the data is already sorted
    
    Returns
    D    -- float: KS test statistic
    pval -- float: p-value to reject null hypothesis 
        
    """
    N1 = len(x1)
    N2 = len(x2)
    
    if not sorted: # Sort data in ascending order to approximate the ECDF
        # quicksort is performed in place, so copy data first
        x1 = quicksort(np.copy(x1))
        x2 = quicksort(np.copy(x2))

    # empirical cdf of both functions as function of their datapoints
    ECDF1 = sr.linspace(1,N1,N1)/N1
    ECDF2 = sr.linspace(1,N2,N2)/N2
    
    D = 0 # Maximum distance between ECDFs
    i, j = 0, 0
    while (i < (N1-1)) and (j < (N2-1)): # Loop through both arrays
        # CDF of x1 is to the left of x2, increment x1
        while (x1[i] <= x2[j]) and (i < (N1-1)):
            i += 1
        # CDF of x2 is to the left of x1, increment x2
        while (x2[j] <= x1[i]) and (j < (N2-1)):
            j += 1
        # Distance between where we currently are in the CDFs
        dist = np.abs(ECDF1[i]-ECDF2[j]) 
        if dist > D:
            D = dist
    
    sqN = np.sqrt(N1*N2/(N1+N2))
    z = (sqN+0.12+0.11/sqN)*D
    pval = 1-KScdf(z)
    
    return D, pval
    
def KuiperTest_2sample(x1, x2, sorted=False):
    """
    Compute the 2 sample Kuiper Test on two datasets
    
    x1     -- array:    realisation from some PDF 
    x2     -- array:    realisation from some PDF
    sorted -- boolean:  whether the data is already sorted
    
    Returns
    V    -- float: Kuiper test statistic
    pval -- float: p-value to reject null hypothesis 
        
    """        
    N1 = len(x1)
    N2 = len(x2)
    
    if not sorted: # Sort data in ascending order to approximate the ECDF
        # quicksort is performed in place, so copy data first
        x1 = quicksort(np.copy(x1))
        x2 = quicksort(np.copy(x2))

    # empirical cdf of both functions as function of their datapoints
    ECDF1 = sr.linspace(1,N1,N1)/N1
    ECDF2 = sr.linspace(1,N2,N2)/N2
    
    # Maximum distance above and below ECDFs
    Dminus = 0 
    Dplus = 0
    i, j = 0, 0
    while (i < (N1-1)) and (j < (N2-1)): # Loop through both arrays
        # CDF of x1 is to the left of x2, increment x1
        while (x1[i] <= x2[j]) and (i < (N1-1)):
            i += 1
        # CDF of x2 is to the left of x1, increment x2
        while (x2[j] <= x1[i]) and (j < (N2-1)):
            j += 1
        # Distance between where we currently are in the CDFs
        distp = ECDF1[i]-ECDF2[j]
        distm = ECDF2[j]-ECDF1[i]
        if distp > Dplus:
            Dplus = distp
        if distm > Dminus:
            Dminus = distm
    
    # Kuiper's statistic
    V = Dplus + Dminus
    
    sqN = np.sqrt(N1*N2/(N1+N2))
    lamb = (sqN+0.155+0.24/sqN)*V
    pval = Kuiperpvalue(lamb)
    
    return V, pval

def quickargsort(arr):
    """
    Argsort with quicksort.
    
    This function works by creating an additional axis to the array
    which saves the indices, and swaps them along with the elements
    
    The subfunction quicksort is equivalent to the 'real' quicksort
    except that it uses some smarter slicing. Note that the original
    input array is not modified in this function
    
    Returns
    indices -- integer array of the indices that would sort the array
    """
    # Store indices to swap them along with the elements
    indices = list(range(0,len(arr)))
    # Make sure to copy arr, since it will get swapped in place
    newarr = np.array([np.copy(arr),indices]).T # shape (len(arr),2)
    
    def quicksort(arr):
        """
        Quicksort adjusted for index sorting
        """
        N = len(arr)

        # Make sure first/last/middle elements are ordered correctly
        if arr[0,0] > arr[N-1,0]: # swap leftmost and rightmost elements
            # Use advanced slicing to swap along rows
            # (since normal slicing now creates views instead of copies)
            arr[[0,N-1],:] = arr[[N-1,0],:]
        if arr[(N-1)//2,0] > arr[N-1,0]: # swap middle and rightmost element
            arr[[(N-1)//2,N-1],:] = arr[[N-1,(N-1)//2],:]
            
        if arr[0,0] > arr[(N-1)//2,0]: # swap middle and leftmost element
            arr[[0,(N-1)//2],:] =  arr[[(N-1)//2,0],:]

        i, j = 0, N-1
        pivot = arr[(N-1)//2,0]
        pivot_position = (N-1)//2
        for _ in range(0,N//2): # Can improve this range()
            while arr[i,0] < pivot:
                i +=  1
            while arr[j,0] > pivot:
                j -= 1
            if j <= i:
                break # pointers have crossed
            else:
                if i == pivot_position: # have to keep track of where the pivot is
                    pivot_position = j # going to switch them
                elif j == pivot_position:
                    pivot_position = i
                # Both i and j found, swap them around the pivot and continue
                arr[[i,j],:] = arr[[j,i],:]

        if N > 2:
            # As long as we don't have 1 element arrays, perform quicksort on the subarrays
            leftarr = arr[:pivot_position] # left of the pivot
            rightarr = arr[pivot_position+1:] # right of the pivot        
            quicksort(leftarr)
            quicksort(rightarr)
    
    # Call the subfunction on the array with the additional axis
    quicksort(newarr)
    # return only the indices
    return np.array(newarr[:,1],dtype='int') 


# In Question 2 we need 1024**2 standard normal random numbers, 
# since we already have a million now, we might as well generate 
# a few more and save them for Question 2
N = 1024
amount = N**2-len(randgauss) # amount we need to generate
new_randnum = np.zeros(amount)
for i in range(amount):
    new_randnum[i] = RNGESUS.get_randomnumber()
all_randnum = np.concatenate([new_randnum,all_randnum])
# transform all random numbers drawn to standard normal dist
randgauss = BoxMuller(all_randnum)
# save these so we can use them later
np.save('./1MrandSN.npy', randgauss)    
# For Question 1, we only need the first 100,000
randgauss = randgauss[:100000]

# Load the data for question e already so we can do question c,d,e
# all in the same loop. # 10 sets of 100,000 random numbers
data = np.loadtxt('./randomnumbers.txt') # shape(int(1e5),10)

# my values
all_D = np.zeros(41)
all_p = np.zeros(41)
# scipy values
all_D_sp = np.zeros(41)
all_p_sp = np.zeros(41)

all_numpoints = np.zeros(41)

# Make the same plot for the Kuipers test
all_D_kuiper = np.zeros(41)
all_p_kuiper = np.zeros(41)
# astropy values
all_D_kuiper_sp = np.zeros(41)
all_p_kuiper_sp = np.zeros(41)

# 2 sample KS test for the datasets
all_D_2s_KS = np.zeros((41,10))
all_p_2s_KS = np.zeros((41,10))
# 2 sample Kuiper's test
all_D_2s_kuiper = np.zeros((41,10))
all_p_2s_kuiper = np.zeros((41,10))

# Calculate CDF of the standard normal once in advance
hCDF = GaussianCdf(randgauss[:100000])

# A spacing of 0.1 dex means we increase by a factor $10^{0.1}$.
# So, to get up to $10^5$ starting from $10^1$, we have to increase 
# by this factor 40 times
for i in range(41):
    numpoints = int(10*10**(0.1*i))
    all_numpoints[i] = (numpoints)

    # Find the index that sorts the current points
    indices = quickargsort(randgauss[:numpoints])
    # Sort the current points
    curpoints = randgauss[:numpoints][indices]    
    # Sort CDF at the points that we will evaluate equivalently
    curCDF = hCDF[:numpoints][indices]
    
    # Tell the KS test that we already sorted the points
    D, pval = KStest(curpoints,curCDF,sorted=True)
    all_D[i] = D
    all_p[i] = pval
    # define a lambda function so scipy works without new calculations
    CDFprecalc = lambda x: curCDF
    D, pval = kstest(curpoints,CDFprecalc)
    all_D_sp[i] = D
    all_p_sp[i] = pval
    
    # Perform the Kuiper test 
    D, pval = KuiperTest(curpoints, curCDF, sorted=True)
    all_D_kuiper[i] = D
    all_p_kuiper[i] = pval
    # Astropy
    D, pval = kuiper(curpoints, CDFprecalc)
    all_D_kuiper_sp[i] = D
    all_p_kuiper_sp[i] = pval
    
    # Compare our random numbers the 10 sets of random numbers too
    for j in range(data.shape[1]):
        curdata = np.copy(data[:numpoints,j])
        # Sort the data, since we have already sorted our own points
        quicksort(curdata)
        # Calculate two sample tests
        D, pval = KStest_2sample(curpoints, curdata, sorted=True)
        all_D_2s_KS[i,j] = D
        all_p_2s_KS[i,j] = pval
        
        D, pval = KuiperTest_2sample(curpoints, curdata, sorted=True)
        all_D_2s_kuiper[i,j] = D
        all_p_2s_kuiper[i,j] = pval   

# Make a plot of the probability that it is consistent with Gaussian
plt.plot(all_numpoints,all_p,'-o')
plt.xlabel('Number of points')
plt.ylabel('$p$-value')
plt.title("KS Test")
plt.xscale('log')
plt.savefig('./plots/q1c1.png')
plt.close()

# Make a plot to compare the statistic with scipy
plt.plot(all_numpoints,all_D,'-o',label='My calculation',alpha=0.5)
plt.plot(all_numpoints,all_D_sp,'-o',label='Scipy',alpha=0.5)
plt.xlabel('Number of points')
plt.ylabel('Statistic D')
plt.title("KS Test")
plt.xscale('log')
plt.legend(frameon=False)
plt.savefig('./plots/q1c2.png')
plt.close()

# Make a plot to compare the p-value with scipy
plt.plot(all_numpoints,all_p,'-o',label='My calculation',alpha=0.5)
plt.plot(all_numpoints,all_p_sp,'-o',label='Scipy',alpha=0.5)
plt.xlabel('Number of points')
plt.ylabel('$p$-value')
plt.title("KS Test")
plt.xscale('log')
plt.legend(frameon=False)
plt.savefig('./plots/q1c3.png')
plt.close()

# #### d) Write a code that does the Kuiper's test on your random numbers and
# make the same plot as for the KS-test.

# Make a plot of the probability that it is consistent with Gaussian
plt.plot(all_numpoints,all_p_kuiper,'-o')
plt.xlabel('Number of points')
plt.ylabel('p-value')
plt.title("Kuiper's Test")
plt.xscale('log')
plt.savefig('./plots/q1d1.png')
plt.close()

# Make a plot to compare the statistic with astropy
plt.plot(all_numpoints,all_D_kuiper,'-o',label='My calculation',alpha=0.5)
plt.plot(all_numpoints,all_D_kuiper_sp,'-o',label='Astropy',alpha=0.5)
plt.xlabel('Number of points')
plt.ylabel('Statistic V')
plt.title("Kuiper's Test")
plt.xscale('log')
plt.legend(frameon=False)
plt.savefig('./plots/q1d2.png')
plt.close()

# Make a plot to compare the p-value with astropy
plt.plot(all_numpoints,all_p_kuiper,'-o',label='My calculation',alpha=0.5)
plt.plot(all_numpoints,all_p_kuiper_sp,'-o',label='Astropy',alpha=0.5)
plt.xlabel('Number of points')
plt.ylabel('p-value')
plt.title("Kuiper's Test")
plt.xscale('log')
plt.legend(frameon=False)
plt.savefig('./plots/q1d3.png')
plt.close()


# #### e) Download a dataset. This dataset contains 10 sets of random numbers. 
# Compare these 10 sets with your Gaussian pseudo random numbers and make the
#  same plot of the probability in as in either of the previous two exercises 
# (your choice). Which random number array(s) is are consistent with standard 
# normal random numbers?

# See above loop for the calculations
for ds in range(data.shape[1]):
    # Make a plot of the probability that the dataset is consistent with Gaussian
    plt.plot(all_numpoints,all_p_2s_KS[:,ds],'-o',label='KS')
    plt.plot(all_numpoints,all_p_2s_kuiper[:,ds],'-o',label='Kuiper')
    plt.xlabel('Number of points')
    plt.ylabel('$p$-value')
    plt.title(f"Set {ds}. Two sample KS test, p={all_p_2s_KS[-1,ds]}     \n Two sample Kuiper test, p={all_p_2s_kuiper[-1,ds]}")
    plt.xscale('log')
    plt.legend(frameon=False)
    plt.savefig(f'./plots/q1e{ds}.png')
    plt.close()