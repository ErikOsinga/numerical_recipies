#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import some_routines as sr


# # 2. Making an initial density field

# We only have to generate the upper half of the fourier plane
# since the lower half will be given by the complex conjugate 
# because the field has to be real. Similary, for ky=0
# we have that opposite kx's will be complex conjugates.



def model_n(k, n):
    """Power spectrum powerlaw model"""
    return k**n

def density_field(N, model, randgauss):
    """
    Generate a real density field with mean 0
    that follows a given power spectrum model.
    
    N         -- int: size of the field
    model     -- Power spectrum model function of k
    randgauss -- N**2 standard normal numbers for quick construction
    """
    
    fftfield = np.zeros((N,N),dtype='complex')
    # One step in k
    dk = 2*np.pi/N 
    # The fourier frequencies are different for even or odd N
    Neven = N%2 # add one to loops if N is odd
    
    counter = 0
    # Loop over all kx modes
    for i in range(0,N): 
        if i <= N//2:
            kx = dk*i
        else:
            kx = (-N+i)*dk
            
        # start at j=1 because we generate the kx's on the 
        # ky=0 -axis seperately. Additionally, only generate the 
        # upper half of the fourier plane (ky>0)
        for j in range(1,N//2+Neven):
            ky = dk*j               
            k = (kx**2+ky**2)**0.5
#             Transform standard normal numbers to correct variance.
#             Since these modes will be conjugated and put into the
#             lower half of the Fourier plane, we have to divide the
#             variance by 2 in order to satisfy total variance being
#             equal to P(k) at k=k
            fftfield[i,j] = randgauss[counter]*(model(k)/2)**0.5 + 1j*(
                            randgauss[counter+1]*(model(k)/2)**0.5)
            counter += 2
    if Neven == 0:
        # We have an even amount of N, so do not forget the N//2
        # column
        ky = N//2*dk
        for i in range(1,N//2):
            kx = dk*i
            k = (kx**2+ky**2)**0.5
            # Note again division by two of the variance.
            fftfield[i,N//2] = randgauss[counter]*(model(k)/2)**0.5 +1j*(
                               randgauss[counter+1]*(model(k)/2)**0.5)
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
    for i in range(1,N//2+Neven):
        fftfield[i,0] = randgauss[counter]*(model(k)/2)**0.5 + 1j*(
                            randgauss[counter+1]*(model(k)/2)**0.5)
        counter += 2
        # complex conjugate
        fftfield[-i,0] = fftfield[i,0].real - 1j*fftfield[i,0].imag
        
        
    # Finally generate all modes below the kx axis by conjugating
    # all modes above the kx axis 
    for i in range(0,N):
        for j in range(N//2,N):
            fftfield[i,j] = fftfield[-i,-j].real - 1j*fftfield[-i,-j].imag
    
    # Don't forget that the [0,0] component of the field has to be 0
    # for a GRF with mean 0
    fftfield[0,0] = 0 + 1j*0
    
    # Multiply by N^2 to undo the normalization build into scipy
    densfield = scipy.fftpack.ifft2(fftfield)*N**2
    
    return fftfield, densfield
            
# use the more than 1 million random numbers we generated before
randgauss = np.load('./1MrandSN.npy')

# We can assign our own scale, since the power spectrum is a simple
# power law. Thus we assume here one pixel is one Kpc
for i,n in enumerate([-1,-2,-3]):
    # Power spectrum power law
    model = lambda k: model_n(k,n)
    fftfield, densfield = density_field(1024,model,randgauss)    

    plt.title(f"Density field from $P(k)=k^{ {n} }$")
    # Fix that center x,y = (0,0) in the middle of the plot
    plt.imshow(densfield.real, extent=[-densfield.shape[1]/2
               ,densfield.shape[1]/2, -densfield.shape[0]/2
               ,densfield.shape[0]/2] )
    plt.xlabel('y (kpc)')
    plt.ylabel('x (kpc)')
    cbar = plt.colorbar()
    plt.savefig(f'./plots/q2_{i}.png')
    plt.close()