#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import numpy.fft
import some_routines as sr

# # 4. Zeldovich approximation.

# #### a) Calculate the growth factor at z=50 numerically with a relative accuracy of $10^-5$. 

# Since the function has an integrable singularity at the limit $a=0$ 
# we can use extended midpoint Romberg integration to integrate this function. 
# However, it is simpler to simply evaluate the limit. We show in the latex PDF
# that the limit of the function is zero.
# Thus we can simply set the function to 0 whenever we evaluate the function inside the integral at $a=0$

def Hubbleparam(z, Omega_m, Omega_Lamb):
    """
    Return the Hubble parameter H(z) in terms of H0
    given current cosmological constants 
    """
    return np.sqrt(Omega_m*(1+z)**3+Omega_Lamb)

def integrand(a, Omega_m, Omega_Lamb):
    """
    Return the function inside the integral as a function of a
    """
    # The integrand goes to 0 for a=0, 
    # but python does not like division by zero    
    where0 = a == 0
    a[where0] = 1e10 # Put zeroes temporarily on some large number   
    ans = 1/(Omega_m/a + Omega_Lamb*a**2)**(3/2)
    ans[where0] = 0 # Make sure the function is 0 for a=0
    return ans
    
def growth_factor(a, Omega_m, Omega_Lamb):
    """
    Return the growth factor at given scale factor.
    Uses the functions Hubbleparam() and integrand()
    """
    # Transform from a to z
    z = 1/a - 1
    
    # Function that will be integrated over a
    to_integ = lambda a: integrand(a, Omega_m, Omega_Lamb)

    Dz = 5*Omega_m/2 * Hubbleparam(z,Omega_m,Omega_Lamb)
    # save integ seperately, needed for b)
    integ = sr.romberg(to_integ,0,a,order=6,relacc=1e-5)[0]
    Dz *= integ
    
    return Dz, integ
    
Omega_m, Omega_Lamb = 0.3, 0.7
z = 50
a = 1/(1+z)
# Calculate growth factor at z=50
Dz, integ = growth_factor(a, Omega_m, Omega_Lamb)
print ("Growth factor at z=50:", Dz)


# I is the value of the integral only, needed for b
H0 = 70 # km/s/Mpc
I = integ/H0**3


# #### b) We also want to calculate the derivative at z=50 in order to be able 
# to calculate the momentum of the particles. Calculate the derivative analytically
# and give its value at z=50. Bonus point if you also numerically match the
# analytical result within $10^{-8}$

def analytical_deriv(z):
    """return analytical derivative at z=z"""
    a = 1/(1+z)
    if z == 50: # use earlier computed result
        Inum = I
    else:
        # use Romberg
        to_integ = lambda a: integrand(a, Omega_m, Omega_Lamb)
        integ = sr.romberg(to_integ,0,a,order=6,relacc=1e-5)[0]
        Inum = integ/H0**3
        
    ans = 5*Omega_m*H0**2/(2*a**3*H0
                           *Hubbleparam(z, Omega_m, Omega_Lamb))
    ans *= (-3*Omega_m*Inum*H0**3*Hubbleparam(z,Omega_m,Omega_Lamb)/2 + 1)

    return ans

anderiv = analytical_deriv(z=50)
print (f"Analytical derivative at z=50: {anderiv} s^(-1)")

# #### c) Use the Zeldovich approximation to generate a movie of the evolution 
# of a volume in two dimensions from a scale factor of 0.0025 until a scale 
# factor of 1.0. Use 64x64 particles in a square grid. Your movie should contain
# at least 30 frames per seconds and should at least last 3 seconds. 
# Also plot the position and momentum of the first 10 particles along the y-direction vs a.

def model_n(k, n):
    """Power spectrum powerlaw model"""
    return k**n

def kvector(N,ndim):
    """
    Generate NxN(xN)xndim matrix of k vector values
    Since we need to do the IFFT of k*c_k
    This kvector array comes in handy to do this
    """
    dk = 2*np.pi/N
    ks = np.zeros(N) # aranged vector of kx modes
    # Loop over all kx modes
    for i in range(0,N): 
        if i <= N//2:
            ks[i] = dk*i
        else:
            ks[i] = (-N+i)*dk
            
    # My implementation of the c_field has a different definition
    # for the x axis than numpy, thus swap y and x from np.meshgrid
    if ndim == 2:
        # every particle has a 2D position
        kvector = np.zeros((N,N,ndim))
        # simply replaces more of the same for loops
        ky, kx = np.meshgrid(ks,ks) # construct a grid
        kvector[:,:,0] = kx
        kvector[:,:,1] = ky
    elif ndim == 3:
        # every particle has a 3D position
        kvector = np.zeros((N,N,N,ndim))
        ky, kx, kz = np.meshgrid(ks,ks,ks)
        kvector[:,:,:,0] = kx
        kvector[:,:,:,1] = ky
        kvector[:,:,:,2] = kz

    return kvector

def qvector(N, ndim):
    """
    Generate NxN(xN)xndim matrix of q vector values
    """
    xpos = sr.linspace(0,N-1,N)
    if ndim == 2:
        qvector = np.zeros((N,N,2))
        ypos, xpos = np.meshgrid(xpos,xpos)
        qvector[:,:,0] = xpos
        qvector[:,:,1] = ypos
    elif ndim == 3:
        qvector = np.zeros((N,N,N,3))
        ypos, xpos, zpos = np.meshgrid(xpos,xpos, xpos)
        qvector[:,:,:,0] = xpos
        qvector[:,:,:,1] = ypos
        qvector[:,:,:,2] = zpos
    
    return qvector
            

def c_field(N, model, randgauss):
    """
    Generate the Fourier space of a real density field with mean 0
    that follows a given power spectrum model.
    Very similar to code in exercise 2, except that 
    the fourier modes are now generated with
    c_k = (ak - ibk)/2
        
    N         -- int: size of the field
    model     -- Power spectrum model function of k
    randgauss -- N**2 standard normal numbers for quick construction
    """
    
    fftfield = np.zeros((N,N),dtype='complex')
    # One step in k
    dk = 2*np.pi/N 
    # The fourier frequencies are different for (un)even N
    Neven = N%2 # add one to loops if N is uneven
    
    counter = 0
    # Loop over all kx modes
    for i in range(0,N): 
        if i <= N//2:
            kx = dk*i
        else:
            kx = (-N+i)*dk
            
        # start at j=1 because we generate the kx's on the 
        # ky-axis seperately. Additionally, only generate the 
        # upper half of the fourier plane (ky>0)
        for j in range(1,N//2+Neven):
            ky = dk*j               
            k = (kx**2+ky**2)**0.5
#             Transform standard normal numbers to correct variance.
#             Since these modes will be conjugated and put into the
#             lower half of the Fourier plane, we have to divide the
#             variance by 2 in order to satisfy total variance being
#             equal to P(k) at k=k
                                                # Note the - and /2
            fftfield[i,j] = (randgauss[counter]*(model(k)/2)**0.5 - 1j*(
                            randgauss[counter+1]*(model(k)/2)**0.5) )/2
            counter += 2
    if Neven == 0:
        # We have an even amount of N, so do not forget the N//2
        # column
        ky = N//2*dk
        for i in range(1,N//2):
            kx = dk*i            
            k = (kx**2+ky**2)**0.5
            # Note again division by two of the variance.
            # Note now also the - and /2 
            fftfield[i,N//2] = (randgauss[counter]*(model(k)/2)**0.5 -1j*(
                               randgauss[counter+1]*(model(k)/2)**0.5))/2
            counter += 2
            # Complex conjugate
            fftfield[-i,N//2] = fftfield[i,N//2].real - 1j*(
                                fftfield[i,N//2].imag)
            
        # Now some numbers are their own complex conjugate.
        # i.e., they are real. No dividing by two of the variance.
        k = (N//2*dk)
        fftfield[0,N//2] = randgauss[counter+1]*model(k)**0.5 + 1j*0
        fftfield[N//2,0] = randgauss[counter+2]*model(k)**0.5 + 1j*0
        k *= np.sqrt(2)
        fftfield[N//2,N//2] = randgauss[counter+3]*model(k)**0.5 + 1j*0
        counter += 3
        
    # The kx-axis is conjugate symmetric in kx
    # so we only have to generate half of this axis 
    for i in range(1,N//2+Neven):                         # - and /2
        fftfield[i,0] = (randgauss[counter]*(model(k)/2)**0.5 - 1j*(
                            randgauss[counter+1]*(model(k)/2)**0.5))/2
        counter += 2
        # complex conjugate
        fftfield[-i,0] = fftfield[i,0].real - 1j*fftfield[i,0].imag
        
        
    # Finally generate all modes below the kx axis by conjugating
    # all modes above the kx axis 
    for i in range(0,N):
        for j in range(N//2,N):
            fftfield[i,j] = fftfield[-i,-j].real - 1j*fftfield[-i,-j].imag
    
    # Don't forget that the [0,0] component of the field has to be 0
    fftfield[0,0] = 0 + 1j*0   
    
    return fftfield


# We have to divide the power spectrum by k**2. So might as well put this into
# the power spectrum

# Divide the power spectrum by another factor 10, just normalization
# (as suggested by Folkert to make the video easier to watch)
Peff = lambda k: model_n(k, -2)/(10*k**4)

# Reuse the random numbers we had earlier
randgauss = np.load('./1MrandSN.npy')

N = 64
cfield = c_field(N,Peff,randgauss)    
kvec = kvector(N,2)

# Calculate displacement vector. Equation 9 in Handin, seperated per dimension.
# first dimension 
Sfield0 = numpy.fft.ifft2(1j*cfield*kvec[:,:,0])*N**2
# second dimension
Sfield1 = numpy.fft.ifft2(1j*cfield*kvec[:,:,1])*N**2
Sfield = np.zeros((N,N,2))
Sfield[:,:,0] = Sfield0.real # x dimension
Sfield[:,:,1] = Sfield1.real # y dimension

def momentum(a,da):
    """
    Calculate momentum of all particles at scale factor a
    Given a and stepsize da
    """
    anow = a-da/2
    znow = 1/anow - 1
    
    # momentum of all particles, uses analytical derivative, which uses
    # the numerical romberg integration 
    ans = -1*anow**2 * analytical_deriv(znow)*Sfield # (64,64,2)
                                                # or (64,64,64,3)
    return ans

# Initial positions, 64*64*2 array 
# containing x,y pos of all 64x64 particles
qvec = qvector(N, 2)

# a = 0.0025 is the initial timestep 
# We need at least 90 frames, so let's do 100
all_a = sr.linspace(0.0025,1,100)
da = all_a[1]-all_a[0]
# positions of first 10 particles along y direction
first_positions = np.zeros((len(all_a),10,2))
# momentum of first 10 particles along y direction
first_momenta = np.zeros((len(all_a),10,2))

for i, a in enumerate(all_a):
    xvec = qvec + growth_factor(a, Omega_m, Omega_Lamb)[0]*Sfield
    # Make sure we have periodic boundary conditions
    xvec %= N
    
    # Save first 10 particle positions
    first_positions[i,:,:] = xvec[0,:10,:]
    # Save first 10 particle momenta
    if i>0:
        # keep only first 10
        p = momentum(a, da)[0,:10,:]
        first_momenta[i,:] = p
    
    plt.title(f'a={a}')
    plt.scatter(xvec[:,:,0],xvec[:,:,1],alpha=0.5)
    plt.xlabel('$x$ (Mpc)')
    plt.ylabel('$y$ (Mpc)')
    plt.savefig(f'./plots/movie/4c_{i:04d}.png')
    plt.close()


# Also plot the position and momentum of the first 10 particles
# along the y-direction vs a

# y-position
for i in range(10):
    # Particles should start around y= 0(=64) to 10
    plt.plot(all_a,first_positions[:,i,1]) # plot only y coordinate
    plt.ylabel('$y$-position')
    plt.xlabel('$a$')
plt.savefig('./plots/q4c1.png')
plt.close()

# y-momentum
for i in range(10):
    plt.plot(all_a,first_momenta[:,i,1]) # plot only y coordinate
    # Physical dimension we assigned was Mpc. 
    plt.ylabel('$y$-momentum per unit mass (Mpc/s)')
    plt.xlabel('$a$')
plt.savefig('./plots/q4c2.png')
plt.close()

# #### d) Generate initial conditions for a 3D box to do an N-body simulation, 
# make initial conditions for $64^3$ particles starting at redshift 50. 
# Besides this make 3 seperate movies of a slice of thickness 1/64th of your box
# at its center, make a slice for x-y, x-z and y-z. 
# Again make a movie of at least 3 seconds with at least 30 frames per second. 
# Remember: slice, not projection. 
# Finally, plot the position and momentum of the first 10 particles along the
# z-direction vs a

def c_k_array(N, model, randgauss, counter=0):
    """
    Generate NxNxN matrix of C_k values at every kx,ky_kz combination
    C_k only depends on the modulus of the k vector. 
    This is implicitly used in the function that generates the 2D field
    but is not as simple to implement in the 3D field implementation.
    Therefore we use this function.
    
    N         -- int: size of the field
    model     -- Power spectrum model function of k
    randgauss -- N**3 standard normal numbers for quick construction
    counter   -- which random number to start at
    """
    dk = 2*np.pi/N
    ks = np.zeros(N) # aranged vector of kx modes
    # Loop over all kx modes
    for i in range(0,N): 
        if i <= N//2:
            ks[i] = dk*i
        else:
            ks[i] = (-N+i)*dk  
    # every particle has a 3D position
    kvector = np.zeros((N,N,N,3))
    ky, kx, kz = np.meshgrid(ks,ks,ks)
    # Modulus, (64,64,64) array
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    # Calculate the variance, set k[0,0,0] to 1 to prevent division by 0 
    k[0,0,0] = 1
    # Standard deviation as function of k
    std = np.sqrt(model(k))
    std[0,0,0] = 0 # No contribution
    
    # Multiply random numbers with the std to get a_k and b_k
    a_k = std*randgauss[counter:std.size+counter].reshape(std.shape)
    b_k = std*randgauss[counter+std.size:2*std.size+counter].reshape(std.shape)
    
    # Note the - and the /2
    c_k = (a_k - 1j*b_k)/2 # (64,64,64) 
    return c_k

def c_field3D(N, model, randgauss, counter=0):
    """
    Generate the Fourier space of a real density field with mean 0
    that follows a given power spectrum model.
    The density field is generated in 3D by generating half the 3D cube and then
    making the other half of the cube (with ky<0) the complex conjugate
    
    N         -- int: size of the field
    model     -- Power spectrum model function of k
    randgauss -- N**3 standard normal numbers for quick construction
    counter   -- which random number to start at
    """
    
    fftfield = np.zeros((N,N,N),dtype='complex')
    # All random numbers we will ever need
    c_k = c_k_array(N, model, randgauss, counter)
    # One step in k
    dk = 2*np.pi/N 
    # The fourier frequencies are different for (un)even N
    Neven = N%2 # add one to loops if N is uneven
    
    # Loop over all kz modes
    for z in range(0,N):
        # Loop over all kx modes
        for i in range(0,N): 
            # start at j=1 because we generate the kx's and kz's on the 
            # ky-axis seperately. Additionally, only generate the 
            # half of the fourier cube (ky>0)
            for j in range(1,N//2+Neven):
                # Use earlier computed c_k values
                fftfield[i,j,z] = c_k[i,j,z]
                
    if Neven == 0:
        # We have an even amount of N, so do not forget the j = N//2
        # plane
        for z in range(1,N//2):
            for i in range(1,N//2):
                fftfield[i,N//2,z] = c_k[i,j,z]
                # Complex conjugate
                fftfield[-i,N//2,-z] = fftfield[i,N//2,z].real - 1j*(
                                    fftfield[i,N//2,z].imag)

        # Now some numbers are their own complex conjugate.
        # i.e., they are real.
        fftfield[0, 0, N//2] = c_k[0,0,N//2].real
        fftfield[0, N//2, 0] = c_k[0,N//2,0].real
        fftfield[N//2, 0, 0] = c_k[N//2, 0, 0].real
        
        fftfield[0, N//2, N//2] = c_k[0,N//2,N//2].real
        fftfield[N//2, N//2, 0] = c_k[N//2,N//2,0].real
        fftfield[N//2, 0, N//2] = c_k[N//2, 0, N//2].real
        
        fftfield[N//2, N//2, N//2] = c_k[N//2, N//2, N//2].real
        
    # The ky=0 plane is conjugate symmetric in kx,kz
    # so we only have to generate half of this plane
    for z in range(1,N//2+Neven):
        for i in range(1,N//2+Neven):
            fftfield[i,0,z] = c_k[i,0,z]
            # complex conjugate
            fftfield[-i,0,-z] = c_k[i,0,z].real - 1j*c_k[i,0,z].imag

    # Finally generate all modes with ky>0 by conjugating
    # all modes with ky < 0 
    for z in range(0,N):
        for i in range(0,N):
            for j in range(N//2,N):
                fftfield[i,j,z] = fftfield[-i,-j,-z].real - 1j*fftfield[-i,-j,-z].imag
    
    # Don't forget that the [0,0] component of the field has to be 0
    fftfield[0,0,0] = 0 + 1j*0   
    
    return fftfield


# Now we repeat basically the same thing, except we start at z=50
a = 1/51
kvec = kvector(N,3)

Peff = lambda k: model_n(k, -2)/(100*k**4)
cfield = c_field3D(N, Peff, randgauss)

# Equation 9 again seperated per dimension
Sfield0 = numpy.fft.ifftn(1j*cfield*kvec[:,:,:,0])*N**3
Sfield1 = numpy.fft.ifftn(1j*cfield*kvec[:,:,:,1])*N**3
Sfield2 = numpy.fft.ifftn(1j*cfield*kvec[:,:,:,2])*N**3

Sfield = np.zeros((N,N,N,3))
Sfield[:,:,:,0] = Sfield0.real # x dimension
Sfield[:,:,:,1] = Sfield1.real # y dimension
Sfield[:,:,:,2] = Sfield2.real # z dimension

# 3D initial grid positions
qvec = qvector(N, 3)

all_a = sr.linspace(a,1,100)
da = all_a[1]-all_a[0]
# positions of first 10 particles along z direction
first_positions = np.zeros((len(all_a),10,3))
# momentum of first 10 particles along z direction
first_momenta = np.zeros((len(all_a),10,3))

for i, a in enumerate(all_a):
    xvec = qvec + growth_factor(a, Omega_m, Omega_Lamb)[0]*Sfield
    # Make sure we have periodic boundary conditions
    xvec %= N
    
    # Save first 10 positions along z direction
    first_positions[i,:,:] = xvec[0,0,:10,:]
    if i>0:
        # Save first 10 momenta
        p = momentum(a, da)[0,0,:10,:]
        first_momenta[i,:] = p
    
    plt.title(f'a={a}')
    # slice in z of thickness 1 at the center, plot x-y
    mask = ((xvec[:,:,:,2] > 31) & (xvec[:,:,:,2] < 32))
    plt.scatter(xvec[:,:,:,0][mask],xvec[:,:,:,1][mask],alpha=0.5)
    plt.xlabel('$x$ (Mpc)')
    plt.ylabel('$y$ (Mpc)')
    plt.savefig(f'./plots/movie/4d_xy{i:04d}.png')
    plt.close()
    
    plt.title(f'a={a}')
    # slice in y at the center, plot x-z
    mask = ((xvec[:,:,:,1] > 31) & (xvec[:,:,:,1] < 32))
    plt.scatter(xvec[:,:,:,0][mask],xvec[:,:,:,2][mask],alpha=0.5)
    plt.xlabel('$x$ (Mpc)')
    plt.ylabel('$z$ (Mpc)')
    plt.savefig(f'./plots/movie/4d_xz{i:04d}.png')
    plt.close()
    
        
    plt.title(f'a={a}')
    # slice in x at the center, plot y-z
    mask = ((xvec[:,:,:,0] > 31) & (xvec[:,:,:,0] < 32))
    plt.scatter(xvec[:,:,:,1][mask],xvec[:,:,:,2][mask],alpha=0.5)
    plt.xlabel('$y$ (Mpc)')
    plt.ylabel('$z$ (Mpc)')
    plt.savefig(f'./plots/movie/4d_yz{i:04d}.png')
    plt.close()

# Also plot the position and momentum of the first 10 particles
# along the z-direction vs a

# z-position
for i in range(10):
    plt.plot(all_a,first_positions[:,i,1]) # plot only y coordinate
    plt.ylabel('$z$-position')
    plt.xlabel('$a$')
plt.savefig('./plots/q4d1.png')
plt.close()

# y-momentum
for i in range(10):
    plt.plot(all_a,first_momenta[:,i,1]) # plot only y coordinate
    # Physical dimension we assigned was Mpc. 
    plt.ylabel('$z$-momentum per unit mass (Mpc/s)')
    plt.xlabel('$a$')
plt.savefig('./plots/q4d2.png')
plt.close()
