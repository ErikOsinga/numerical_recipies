#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

import some_routines as sr

# # 5. Mass assignment schemes
# 
# #### a). NGP

class NGP(object):
    def __init__(self, nperdim, positions):
        """
        nperdim   -- int: amount of points per dimension
        positions -- array: 3D positions of the particles (3,?)
        """
        self.nperdim = nperdim
        # Particle positions [xvalues,yvalues,zvalues]
        self.positions = positions
        # cell size
        self.deltax = 1
        # Masses of the gridpoints
        self.massgrid = self.assign_mass()
        
    def assign_mass(self):
        """
        Loop over all particles and assign their mass to the
        nearest gridpoint
        """
        massgrid = np.zeros((self.nperdim,self.nperdim,self.nperdim))
        # Index of the cell containing the particle is floored int
        indices = np.array(self.positions, dtype='int')
        # Assign each particle mass to its nearest gridpoint
        for i in range(indices.shape[1]):
            massgrid[tuple(indices[:,i])] += 1
        return massgrid

# Particle positions
np.random.seed(121)
positions = np.random.uniform(low=0,high=16,size=(3,1024))

# Construct the 16^3 grid and assign mass according to NGP
ngp = NGP(16,positions)

# Plot x-y slices
for i, z in enumerate([4,9,11,14]):
    plt.title(f"z={z}")
    # Fix the imshow axes with extent
    plt.imshow(ngp.massgrid[:,:,z],extent=[0,16,16,0])
    cbar = plt.colorbar()
    cbar.set_label('Mass')
    # remember switch x and y in arrays
    plt.xlabel('y')
    plt.ylabel('x')
    plt.savefig(f'./plots/q5a_{i}.png')
    plt.close()


# #### b) To check the robustness of your implementation make a plot of the $x$ 
# position of an individual particle and the value in cell 4 in 1 dimension. 
# Let $x$ vary from the lowest value to the highest possible value in $x$. 
# Repeat for cell 0

all_x = sr.linspace(0,16,34)
cell4 = np.zeros(34)
cell0 = np.zeros(34)
for i, x in enumerate(all_x):
    x = x%16 # periodic boundary conditions
    testpos = np.array([x,0,0])[:,np.newaxis] 
    ngp = NGP(16,testpos)
    cell4[i] = ngp.massgrid[4,0,0]
    cell0[i] = ngp.massgrid[0,0,0]
    
# Should peak between x = 4 and 5
plt.plot(all_x,cell4,'o-')
plt.ylabel('Mass in cell 4')
plt.xlabel('x-position of particle')
plt.savefig(f'./plots/q5b1.png')
plt.close()

# Should peak between x = 0(=16) and 1
plt.plot(all_x,cell0,'o-')
plt.ylabel('Mass in cell 0')
plt.xlabel('x-position of particle')
plt.savefig(f'./plots/q5b2.png')
plt.close()


# #### c) CIC
# 

class CIC(object):
    def __init__(self, nperdim, positions):
        """
        nperdim   -- int: amount of gridpoints per dimension
        positions -- array: 3D positions of the particles (3,?)
        """
        self.nperdim = nperdim
        # Particle positions [xvalues,yvalues,zvalues]
        self.positions = positions
        # cell size
        self.deltax = 1
        # Masses of the gridpoints
        self.massgrid = self.assign_mass()
        
    def assign_mass(self):
        """
        Use the CIC method to assign mass to all the cells.
        Loop over all particles.
        """
        massgrid = np.zeros((self.nperdim,self.nperdim,self.nperdim))
        # Index of the cell containing the particle is floored int
        indices = np.array(self.positions, dtype='int')
        # Distances to nearest grid point in all directions
        # The center of cell 'ijk' is ijk+0.5. 
        # (e.g., (0.5,0.5,0.5_ for cell 0)
        dr = self.positions-(indices+0.5)
        # Check if neighbour is the next or previous cell
        # If the distance is positive, we need to add mass to
        # the next cell. If the distance is negative, to the previous
        # Thus, next index is the sign of the distance
        next_idx = np.array(np.sign(dr),dtype='int')    
        # Then make sure, as we know, distances are always positive
        dr = np.abs(dr)
        tr = 1-dr
        
        # Assign each particle mass to 8 gridpoints
        for i in range(indices.shape[1]):
            indx = indices[:,i]
            # Current cell
            massgrid[tuple(indx)] += np.prod(tr[:,i])
            # Neighbours. Note periodic boundary conditions
            nb = next_idx[:,0]
            
            idx = tuple((indx+np.array([nb[0],0,0]))%self.nperdim)
            massgrid[idx] += dr[0,i]*tr[1,i]*tr[2,i]
            idx = tuple((indx+np.array([0,nb[1],0]))%self.nperdim)
            massgrid[idx] += tr[0,i]*dr[1,i]*tr[2,i]
            idx = tuple((indx+np.array([nb[0],nb[1],0]))%self.nperdim)
            massgrid[idx] += dr[0,i]*dr[1,i]*tr[2,i]
            idx = tuple((indx+np.array([0,0,nb[2]]))%self.nperdim)
            massgrid[idx] += tr[0,i]*tr[1,i]*dr[2,i]
            idx = tuple((indx+np.array([nb[0],0,nb[2]]))%self.nperdim)
            massgrid[idx] += dr[0,i]*tr[1,i]*dr[2,i]
            idx = tuple((indx+np.array([0,nb[1],nb[2]]))%self.nperdim)
            massgrid[idx] += tr[0,i]*dr[1,i]*dr[2,i]
            idx = tuple((indx+np.array([nb[0],nb[1],nb[2]]))%self.nperdim)
            massgrid[idx] += dr[0,i]*dr[1,i]*dr[2,i]
        return massgrid
    
    def compute_force(self, pindices, potential):
        """
        Return gradient of the potential 
        (= -1*Force for unit mass particles.)
        The gradient of the potential is calculated with inverse CIC
        interpolation. In this way we follow Newton's third law.
        
        pindices  -- list of integers, the indices of the particles
                     to compute the gradient for.
                     --> must be between 0 and len(self.positions)
                     
        potential -- 3D array, the potential of every cell
        
        Returns
        all_gradients -- array of shape (len(pindices),3)
                         containing for every particle the 3D gradient
        """
        # Calculate gradient in x direction of every cell
        # with central difference method
        Fx = (np.roll(potential,-1,axis=0) - np.roll(potential,1,axis=0))/(2*self.deltax)
        # Repeat for y and z direction, shape (Nperdim,Nperdim,Nperdim)
        Fy = (np.roll(potential,-1,axis=1) - np.roll(potential,1,axis=1))/(2*self.deltax)
        Fz = (np.roll(potential,-1,axis=2) - np.roll(potential,1,axis=2))/(2*self.deltax)
        
        gradpot = np.zeros((self.nperdim,self.nperdim,self.nperdim,3))
        gradpot[:,:,:,0] = Fx
        gradpot[:,:,:,1] = Fy
        gradpot[:,:,:,2] = Fz
        
        # Indices of the cells containing the particles is floored int
        indices = np.array(self.positions[:,pindices], dtype='int')

        # Equivalent to mass assignment scheme
        dr = self.positions[:,pindices]-indices
        next_idx = np.array(np.sign(dr),dtype='int')    
        dr = np.abs(dr)
        tr = 1-dr
        
        # Interpolate each of the 8 gridpoints contribution to the
        # single value for the gradient of the potential
        all_gradients = np.zeros((len(pindices),3)) # each particle has 3D gradient 
        for i in range(len(pindices)):
            # parent cell
            indx = indices[:,i]
            grad = gradpot[tuple(indx)]*np.prod(tr[:,i])
            # Neighbours. Note periodic boundary conditions
            nb = next_idx[:,0]
            
            idx = tuple((indx+np.array([nb[0],0,0]))%self.nperdim)
            grad += gradpot[idx]*dr[0,i]*tr[1,i]*tr[2,i]
            
            idx = tuple((indx+np.array([0,nb[1],0]))%self.nperdim)
            grad += gradpot[idx]*tr[0,i]*dr[1,i]*tr[2,i]
            
            idx = tuple((indx+np.array([nb[0],nb[1],0]))%self.nperdim)
            grad += gradpot[idx]*dr[0,i]*dr[1,i]*tr[2,i]
            
            idx = tuple((indx+np.array([0,0,nb[2]]))%self.nperdim)
            grad += gradpot[idx]*tr[0,i]*tr[1,i]*dr[2,i]
            
            idx = tuple((indx+np.array([nb[0],0,nb[2]]))%self.nperdim)
            grad += gradpot[idx]*dr[0,i]*tr[1,i]*dr[2,i]
            
            idx = tuple((indx+np.array([0,nb[1],nb[2]]))%self.nperdim)
            grad += gradpot[idx]*tr[0,i]*dr[1,i]*dr[2,i]
            
            idx = tuple((indx+np.array([nb[0],nb[1],nb[2]]))%self.nperdim)
            grad += gradpot[idx]*dr[0,i]*dr[1,i]*dr[2,i]
            
            all_gradients[i,:] = grad
            
        return all_gradients

cic = CIC(16,positions)

# Plot the x-y slices
for i, z in enumerate([4,9,11,14]):
    plt.title(f"z={z}")
    plt.imshow(cic.massgrid[:,:,z],extent=[0,16,16,0])
    cbar = plt.colorbar()
    cbar.set_label('Mass')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.savefig(f'./plots/q5c_{i}.png')
    plt.close()

# Plot the x position of an individual particle
all_x = sr.linspace(0,16,34)
cell4 = np.zeros(34)
cell0 = np.zeros(34)
for i, x in enumerate(all_x):
    x = x%16 # periodic boundary conditions
    testpos = np.array([x,0,0])[:,np.newaxis] 
    cicc = CIC(16,testpos)
    cell4[i] = cicc.massgrid[4,0,0]
    cell0[i] = cicc.massgrid[0,0,0]
    
# Should peak at x = 4.5, this is the center of cell 4.
plt.plot(all_x,cell4,'o-')
plt.ylabel('Mass in cell 4')
plt.xlabel('x-position of particle')
plt.savefig('./plots/q5c1.png')
plt.close()

# Should peak at x = 0.5, this is the center of cell 0.
plt.plot(all_x,cell0,'o-')
plt.ylabel('Mass in cell 0')
plt.xlabel('x-position of particle')
plt.savefig('./plots/q5c2.png')
plt.close()


# #### d) Write your own FFT algorithm

def FFT1D(x, IFT=False):
    """
    Perform the (i)FFT of input array x using the Cooley-Tukey 
    algorithm. No normalization
    """
    N = len(x)
    # change the factor when doing IFT
    if IFT:
        fi = 1
    else:
        fi = -1
        
    if N > 1:
        # split in even and odd elements
        ffteven = FFT1D(x[0::2])
        fftodd = FFT1D(x[1::2])
        
        # Exploit the period in k, vectorize instead of loop over k
        if N//2-1 == 0:
            k = 0 # prevent division by zero error in linspace
        else:
            k = sr.linspace(0,N//2-1,N//2)
            
        W = np.exp(fi*2j*np.pi*k/N)*fftodd
        return np.concatenate([ffteven + W
                       , ffteven-W])
    else:
        return x
        

def function(x):
    return 2*np.sin(2*np.pi*x)

N = 64 # amount of samples
x = sr.linspace(0,6*np.pi,N)
# sample spacing
dt = 6*np.pi/64
# k vector, positive part only
ks = np.array(sr.linspace(0, (N-1)//2,(N-1)//2 + 1))
ks /= N*dt
                         
fx = function(x)
fk = FFT1D(fx)
fknp = np.fft.fft(fx)


# Symmetric, so we only plot half of the plane
plt.axvline(1.0,ls='dashed',c='k',label='Analytical result')
plt.plot(ks,np.abs(fk)[:N//2],label='This work')
plt.plot(ks,np.abs(fknp)[:N//2],label='Numpy',ls='dashed')
plt.xlabel('k')
plt.ylabel('$|f(k)|$')
plt.legend(frameon=False,loc='upper left')
plt.savefig('./plots/q5d1.png')
plt.close()


# #### e) Generalize your own FFT algorithm to 2 and 3 dimensions 

def FFT2D(x, IFT=False):
    """
    Perform the FFT of 2D input array x 
    by nesting 1D Fourier transforms, see FFT1D(x)
    """
    FFT = np.array(np.zeros(x.shape),dtype=np.complex)
    # Take 1D fourier transform across rows
    for row in range(x.shape[0]):
        FFT[row,:] = FFT1D(x[row,:],IFT)
    # Then take 1D fourier transform across the columns
    for col in range(x.shape[1]):
        FFT[:,col] = FFT1D(FFT[:,col],IFT)
        
    return FFT

def FFT3D(x, IFT=False):
    """
    Perform the FFT of 3D input array x 
    by doing a 2D Fourier transform x.shape[0] times
    and then doing x.shape[1]*x.shape[2] 1D fourier transforms
    """
    FFT = np.array(x,dtype=np.complex)
    # Take 2D fourier transform across first axis
    for axis in range(x.shape[0]):
        FFT[axis,:,:] = FFT2D(FFT[axis,:,:],IFT)
    # Then take 1D fourier transform of the rest
    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            FFT[:,i,j] = FFT1D(FFT[:,i,j],IFT)
            
    return FFT    

def function(x, y):
    return 2*np.sin(1/2*np.pi*(x+y))

xx, yy = np.meshgrid(x,x)
fxy = function(xx,yy)

# Negative part of k vector, for the axis tick labels
ksneg = -1/dt/N*np.array(sr.linspace(0, (N-1)//2+1,(N-1)//2+2))[1:] # * 2*np.pi
kstot = np.concatenate([ks,ksneg[::-1]])

fkk = FFT2D(fxy)
plt.imshow(np.fft.fftshift(np.abs(fkk)))
plt.imshow(np.log10(np.abs(fkk)))
ax = plt.gca()
# Fix axes labels so it shows the frequency 
# Add another zero because matplotlib ignores first tick
# for some reason
labels = [0] + [f'{kstot[i]:.2f}' for i in range(0,70,10)]
ax.set_yticklabels(labels)
ax.set_xticklabels(labels)
plt.xlabel('$k_x$')
plt.ylabel('$k_y$')
cbar = plt.colorbar()
cbar.set_label('$log_{10} |F(k)|$')
plt.savefig('./plots/q5e1.png')
plt.close()

def multivariateGauss(x, mean,covariance):
    """
    Returns the PDF of a multivariate (n-D) Gaussian.
    Assumes the mean is a vector of length n
    and the covariance is a nxn DIAGONAL matrix.
    (if the matrix is not diagonal, the inverse and determininant are not correct)
    """
    n = len(mean)
    # Calculate determinant and inverse of DIAGONAL covariance
    detcov = 1
    invcov = np.copy(covariance)
    for i in range(n):
        detcov *= covariance[i,i]
        invcov[i,i] = invcov[i,i]**-1.
    # assume dot product is elemental math enough that we can use it
    return ( (2*np.pi)**(-n/2) * (detcov)**(-0.5) 
            * np.exp(-0.5* np.dot((x-mean).T,
                    np.dot(invcov,(x-mean))) ) 
           )

mean = np.array([0.,1.,2.])
# diagonal covariance [3,2,1]
covariance = np.array([[3., 0., 0.],[0., 2., 0.],[0., 0., 1.]])

N = 16 # amount of samples per dimension
all_x = sr.linspace(-3,3,N)
# sample spacing
dt = 6/64
# k vector, positive part only
ks = np.array(sr.linspace(0, (N-1)//2,(N-1)//2 + 1))
ks /= N*dt
# Negative part of k vector, for the axis tick labels
ksneg = -1/dt/N*np.array(sr.linspace(0, (N-1)//2+1,(N-1)//2+2))[1:] # * 2*np.pi
kstot = np.concatenate([ks,ksneg[::-1]])

# Calculate the values at the 16^3 points
triGauss = np.zeros((N,N,N))
for i, xc in enumerate(all_x):
    for j, yc in enumerate(all_x):
        for k, zc in enumerate(all_x):
            x = np.array([xc,yc,zc])
            triGauss[i,j,k] = multivariateGauss(x, mean, covariance)
            
triGaussFT = FFT3D(triGauss)

plt.title('Slice at half $k_z$')
plt.imshow(np.abs(triGaussFT[:,:,8]))
cbar = plt.colorbar()
plt.xlabel('$k_y$')
plt.ylabel('$k_x$')
# Fix axes labels so it shows the frequency 
labels = [0] + [f'{kstot[i]:.2f}' for i in range(0,N,2)]
ax = plt.gca()
ax.set_yticklabels(labels)
ax.set_xticklabels(labels)
plt.savefig('./plots/q5e2.png')
plt.close()

plt.title('Slice at half $k_y$')
plt.imshow(np.abs(triGaussFT[:,8,:]))
cbar = plt.colorbar()
plt.xlabel('$k_z$')
plt.ylabel('$k_x$')
# Fix axes labels so it shows the frequency 
ax = plt.gca()
labels = [0] + [f'{kstot[i]:.2f}' for i in range(0,N,2)]
ax.set_yticklabels(labels)
ax.set_xticklabels(labels)
plt.savefig('./plots/q5e3.png')
plt.close()


plt.title('Slice at half $k_x$')
plt.imshow(np.abs(triGaussFT[8,:,:]))
cbar = plt.colorbar()
plt.xlabel('$k_z$')
plt.ylabel('$k_y$')
# Fix axes labels so it shows the frequency 
ax = plt.gca()
labels = [0] + [f'{kstot[i]:.2f}' for i in range(0,N,2)]
ax.set_yticklabels(labels)
ax.set_xticklabels(labels)
plt.savefig('./plots/q5e4.png')
plt.close()


# #### f) Calculate the potential up to a constant for the same particles. 
# Again make a plot of the potential of $x-y$ slices with $z$ values of 4,9,11 and 14.
# Also make a centered slice for the $x-z$ and $y-z$ plane.

def kvecsquared(N, ndim):
    """
    Generate NxN(xN) matrix of k**2 values to multiply with a grid
    """
    dk = 2*np.pi/N
    ks = np.zeros(N) # aranged vector of kx modes
    # Loop over all kx modes
    for i in range(0,N): 
        if i <= N//2:
            ks[i] = dk*i
        else:
            ks[i] = (-N+i)*dk
            
    # My implementation has a different definition
    # for the x axis than numpy, thus swap y and x from np.meshgrid
    if ndim == 2:
        # every particle has a 2D position
        kvector = np.zeros((N,N,ndim))
        # simply replaces more of the same for loops
        ky, kx = np.meshgrid(ks,ks) # construct a grid
        ksquared = kx**2 + ky**2
    elif ndim == 3:
        # every particle has a 3D position
        kvector = np.zeros((N,N,N,ndim))
        ky, kx, kz = np.meshgrid(ks,ks,ks)
        ksquared = kx**2 + ky**2 + kz**2

    return ksquared

# reduce the values of the mesh
cic = CIC(16,positions)
cic.massgrid /= np.mean(cic.massgrid)
cic.massgrid -= 1

kvec = kvecsquared(16,3)
# Set [0,0,0] to 1 to prevent division by zero
kvec[0,0,0] = 1 # FT of potential at [0,0,0] is zero anyways

# FT of the potential is FT of the density field divided by k^2
cic.FTdens = FFT3D(cic.massgrid)
cic.FTdens /= kvec
# IFFT to get the potential, don't forget normalization
cic.potential = FFT3D(cic.FTdens,IFT=True)/cic.massgrid.size
# It is a real value, discard imaginary part
cic.potential = cic.potential.real

for i, z in enumerate([4,9,11,14]):
    plt.title(f"z={z}")
    plt.imshow(cic.potential[:,:,z],extent=[0,16,16,0])
    cbar = plt.colorbar()
    cbar.set_label('$\Phi (r)$')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.savefig(f'./plots/q5f_{i}.png')
    plt.close()
    
# Also plot centered x-z and y-z plane
plt.title(f"y={8}")
plt.imshow(cic.potential[:,8,:],extent=[0,16,16,0])
cbar = plt.colorbar()
cbar.set_label('$\Phi (r)$')
plt.xlabel('z')
plt.ylabel('x')
plt.savefig('./plots/q5f1.png')
plt.close()

plt.title(f"x={8}")
plt.imshow(cic.potential[8,:,:],extent=[0,16,16,0])
cbar = plt.colorbar()
cbar.set_label('$\Phi (r)$')
plt.xlabel('z')
plt.ylabel('y')
plt.savefig('./plots/q5f2.png')
plt.close()


# #### g) Use your potential field to calculate the gradient of the potential 
# for the first 10 particles. Output the value of the gradient of the potential 
# for these particles in all 3 spatial coordinates. 
# It is important to realize that when assigning inferred quanities for particles 
# they should be assigned with the same weighting as in the case of assiging
# the mass to the grid. Again we use CIC.



# Calculate the gradient of the potential for the first 10 particles
# using inverse CIC to interpolate the gradients of neighbouring cells
pindices = np.array([0,1,2,3,4,5,6,7,8,9])
forces = cic.compute_force(pindices, cic.potential)
print ("Gradient of the potential of first 10 particles:")
print ("Columns denote x,y and z direction respectively")
print (forces)