import numpy as np
import matplotlib.pyplot as plt
from some_routines import linspace

seed = 19231923

def log_kfac(k):
    """
    Calculate the logarithm of the factorial of k
    """
    if k == 0:
        return 0

    ans = np.log(k)
    for x in range(2,int(k)):
        ans += np.log(x)
    return ans

def poisson_probability(k,lambd):
    """
    Returns the Poisson probability, i.e., the probability of 'k' 
    occurences if 'lambd' is the occurence rate, per interval. 
    
    k can be a list or np.array
    """

    lambd = np.float64(lambd) # limit memory
        
    if type(k) == np.ndarray or type(k) == list:
        k = np.asarray(k,dtype='float64') # limit memory
        
        logkfac = []
        for i in range(0,len(k)): # iterate over the array-like object
            logkfac.append(log_kfac(k[i]))
    else:
        k = np.float64(k) # limit memory, assume k is now a float
        logkfac = log_kfac(k)
    
    return np.exp(k*np.log(lambd) - logkfac - lambd)

class RandomGenerator(object):
    """
    Random number generator should be an object because it maintains
    an internal state between calls.
    """
    def __init__(self, seed):
        # make sure the everyhing is an unsigned 64 bit integer
        dtyp = np.uint64
        # the seed for the LGC
        self.X1 = dtyp(seed)
        # the seed for the XORshift
        self.X2 = dtyp(seed)
        
        # Maximum value for an unsigned 64 bit integer
        self.max_value = dtyp(2**64 - 1)
        
        # LCG values from Numerical Recipes
        self.a = dtyp(1664525)
        self.c = dtyp(1013904223)
        self.m = dtyp(2**32)
        
        # 64 bit XOR shift values from Numerical Recipes
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


if __name__ == "__main__":
    print (f"User seed is set to {seed}")

    # Output P_{\lambda}(k) to at least 6 significant digits for these values
    lambdas = [1,5,3,2.6]
    ks = [0,10,20,40]

    for k, lambd in zip(ks, lambdas):
        print(f'P_{lambd}({k}) = {poisson_probability(k,lambd):.5e}')
             
    print ("For the bonus:")   
    lambd, k = 101, 200
    print(f'P_{lambd}({k}) = {poisson_probability(k,lambd):.5e}')

    RNGESUS = RandomGenerator(seed=seed)

    # Generate 1000 random numbers. Plot comparison
    all_randnum = []
    for i in range(1000):
        all_randnum.append(RNGESUS.get_randomnumber())
        
    plt.plot(all_randnum,np.roll(all_randnum,1),'o',alpha=0.5)
    plt.title(f'Comparison of {len(all_randnum)} random numbers')
    plt.xlabel('Value element n')
    plt.ylabel('Value element n+1')
    plt.savefig('./plots/q1b1.png')
    plt.close()

    # Now generate 1 million random numbers. Plot histogram
    all_randnum = []
    for i in range(int(1e6)):
        all_randnum.append(RNGESUS.get_randomnumber())
        
    plt.hist(all_randnum,bins=linspace(0,1,21),edgecolor='black')
    plt.title(f'Histogram of 1 million random numbers')
    plt.xlabel('Value')
    plt.ylabel('Counts')
    plt.savefig('./plots/q1b2.png')
    plt.close()