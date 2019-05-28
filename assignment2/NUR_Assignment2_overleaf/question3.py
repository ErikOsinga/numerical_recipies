#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import some_routines as sr


# # 3. Linear structure growth

# We use the 4-th order Runge-Kutta method to solve the two first order ODEs numerically. 

def f1(z1, z2, t):
    """returns dz1/dt = dD/dt"""
    return z2

def f2(z1, z2, t):
    """returns dz2/dt = d^2D/dt^2"""
    return 2/(3*t**2)*z1 - 4/(3*t)*z2

def rungekutta(func1, func2, h, endpoint, initial_t, initial_y
               ,initial_y1):
    """
    Solve a second order ODE with 4th order Runge Kutta
    
    func1      -- function that returns dz1/dt given y
    func2      -- function that returns dz2/dt given y
    h          -- stepsize
    endpoint   -- end point 
    initial_t  -- initial value of dependent parameter
    initial_y  -- initial value of function
    initial_y1  -- initial value of derivative of function
    
    """
    t = [initial_t]
    y = [initial_y]
    y1 = [initial_y1] 
        
    amount_steps = int((endpoint-initial_t)/h + 0.5)
    for i in range(0,amount_steps):
        # Four function evaluations to evaluate the new position
        # And four function evaluations to evaluate the new derivative
        k11 = h*func1(y[-1],y1[-1],t[-1]) 
        k21 = h*func2(y[-1],y1[-1],t[-1])
        k12 = h*func1(y[-1]+0.5*k11,y1[-1]+0.5*k21,t[-1]+0.5*h)
        k22 = h*func2(y[-1]+0.5*k11,y1[-1]+0.5*k21,t[-1]+0.5*h)
        k13 = h*func1(y[-1]+0.5*k12,y1[-1]+0.5*k22,t[-1]+0.5*h)
        k23 = h*func2(y[-1]+0.5*k12,y1[-1]+0.5*k22,t[-1]+0.5*h)
        k14 = h*func1(y[-1]+0.5*k13,y1[-1]+0.5*k23,t[-1]+h)
        k24 = h*func2(y[-1]+0.5*k13,y1[-1]+0.5*k23,t[-1]+h)
        
        y.append(y[-1]+(k11+2*k12+2*k13+k14)/6)
        y1.append(y1[-1]+(k21+2*k22+2*k23+k24)/6)   
        t.append(t[-1]+h)

        
    return np.array(t), np.array(y), np.array(y1)
    
def analyticalD(D1,Ddt1,t):
    """
    Given initial conditions D(1) and D'(1), returns the analytical
    solution of the linearized density growth equation evaluated at 't'
    """
    A = 3/5 * (D1+Ddt1)
    B = D1 - A
    return A*t**(2/3) + B*t**(-1)

# Initial values
all_D = [3,10,5]
all_dD = [2,-10,0]

# For Case 1, B=0, so there is only a growing mode.
# For Case 2, A=0, so there is only a decaying mode
# For Case 3, Both modes are present
for i in range(3):
    t, D, dDdt = rungekutta(f1,f2,0.001,1000,1,all_D[i],all_dD[i])
    plt.loglog(t,analyticalD(all_D[i],all_dD[i],t)
               ,alpha=0.5,label='Analytical')
    plt.loglog(t,D
               ,alpha=0.5,label='Runge-Kutta',ls='dashed')
    plt.legend(frameon=False)
    plt.xlabel('$t$',fontsize=14)
    plt.ylabel('$D(t)$',fontsize=14)
    plt.title(f"Case {i+1}")
    plt.savefig(f'./plots/q3_{i}.png')
    plt.close()