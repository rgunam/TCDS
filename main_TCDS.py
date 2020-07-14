# -*- coding: utf-8 -*-
"""
Created on February 2020

@author: Dr. Roberto G. Ramírez Chavarría (RRamirezC@iingen.unam.mx)
         Instituto de Ingeniería, Universidad Nacional Autónoma de México
"""

######################################################################################
# This python script includes all methods for time-constant-domain spectroscopy (TCDS) 
# for computing solutions as shown in the paper 
#
# "R.G. Ramírez-Chavarría, C.Sánchez-Pérez, L. Romero-Ornelas and E. Ramón-Gallegos,
#  Time-Constant-Domain Spectroscopy: An Impedance-based Method for Sensing Biological
#  Cells in Suspension, submitted to IEEE Sensors Journal, 2020"
#                                                                                    
# If you make use of it, please cite the paper for academic purposes.   
######################################################################################




import numpy as np
import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt
from math import  ceil, floor
from scipy.integrate import quad
from scipy.linalg import toeplitz



## ----------********************************************************--------------------
## ----------********************************************************--------------------
                        ## TCDS basis function preparation
## ----------********************************************************--------------------
## ----------********************************************************--------------------

def gaussian(t):
    g = lambda x: np.exp(-(x)**2)-1/2
    return g(t)

# Integration algorithm
# Imaginary component
def integration_Z_im(t,epsilon,f_R,f_C):
    tmp = 2*np.pi*f_R/f_C
    integrand = lambda x: tmp/(1./np.exp(x)+(tmp**2)*np.exp(x))*(np.exp(-(epsilon*x)**2))
    out_integral = quad(integrand,-np.inf,+np.inf)
    return out_integral

# Real component
def integration_Z_re(t,epsilon,f_R,f_C):
    tmp = 2*np.pi*f_R/f_C
    integrand = lambda x: (1./(1+(tmp**2)*np.exp(2*x)))*(np.exp(-(epsilon*x)**2))
    out_integral = quad(integrand,-np.inf,+np.inf)
    return out_integral


## ----------********************************************************--------------------
## ----------********************************************************--------------------
                             ## Build regression matrices
## ----------********************************************************--------------------
## ----------********************************************************--------------------
                               
    ###-------------- IMAGINARY COMPONENT Z''_DRT ---------------
def Z_primeprime(tau,freq, epsilon, L):
    R = np.zeros((1,len(freq)))
    C = np.zeros((len(freq),1))
    #out_Z_im = np.zeros((len(freq), 2));
    
    for i in range(0,len(freq)):
        freq_R =  freq[i]
        freq_C = freq[0]
        C[i,0], err = integration_Z_im(tau,epsilon,freq_R,freq_C)
        
    for j in range(0,len(freq)):
        freq_R =  freq[0]
        freq_C = freq[j]
        R[0,j], err = integration_Z_im(tau,epsilon,freq_R,freq_C)
        
        temp_Z_im = toeplitz(C,R)
    
        out_Z_im = np.append(np.zeros((len(freq), 2)), temp_Z_im,axis=1)
        
    return out_Z_im

    ###-------------- REAL COMPONENT Z'_DRT  ---------------
def Z_prime(tau,freq, epsilon, L):
    R = np.zeros((1,len(freq)))
    C = np.zeros((len(freq),1))
    #out_Z_im = np.zeros((len(freq), 2));
    
    for i in range(0,len(freq)):
        freq_R =  freq[i]
        freq_C = freq[0]
        C[i,0], err = integration_Z_re(tau,epsilon,freq_R,freq_C)
        
    for j in range(0,len(freq)):
        freq_R =  freq[0]
        freq_C = freq[j]
        R[0,j], err = integration_Z_re(tau,epsilon,freq_R,freq_C)
        
        temp_Z_re = toeplitz(C,R)
    
        temp_Z_re_2 = np.append(np.ones((len(freq), 1)), temp_Z_re,axis=1)
        out_Z_re = np.append(np.zeros((len(freq), 1)),temp_Z_re_2,axis=1)
        
    return out_Z_re
    

## ----------********************************************************--------------------
## ----------********************************************************--------------------
                      ## Optimization Problem Definition
## ----------********************************************************--------------------
## ----------********************************************************--------------------
def loss_fun(X, Y, beta):
    return cp.pnorm(X @ beta - Y, p=2)**2

def regularizer(beta):
    return cp.pnorm(beta, p=2)**2

def objective_fun(X, Y, beta, lambd):
    return loss_fun(X, Y, beta) + lambd * regularizer(beta)

def mse(X, Y, beta):
    return (1.0 / X.shape[0]) * loss_fun(X, Y, beta).value


## ----------********************************************************--------------------
## ----------********************************************************--------------------
                          ## Interpolate for TCDS
## ----------********************************************************--------------------
## ----------********************************************************--------------------

def interpol_TCDS(freq_out,freq_vec,theta,epsilon):
     out_TCDS = []#np.zeros((1,len(freq_out)))
     g_interp = lambda t,t0: np.exp(-(epsilon*(t-t0))**2)
     x0 = -np.log(freq_vec)
     for k in range(0,len(freq_out)):
         freq_tmp = freq_out[k]
         x = -np.log(freq_tmp)
         out_TCDS.append(np.dot(np.transpose(theta),g_interp(x,x0)))
         TCS_arr = np.array(out_TCDS,dtype=float)
     return TCS_arr



## ----------********************************************************--------------------
## ----------********************************************************--------------------
                          ## Main rutine 
## ----------********************************************************--------------------
## ----------********************************************************--------------------


#______________________________________________________________________________

# 1) ...........   Read experimental data from .csv file 
                        #format: freq,Zreal,Zimag
#______________________________________________________________________________

Z_data = pd.read_csv('RC_circuit_model.csv')

# Convert pandas data frame to numpy array
Z_exp=Z_data.to_numpy()

# Frequency vector
freq_vec = Z_exp[:,0]
N_freqs = len(freq_vec)
tau  = 1/freq_vec

# Z real component
Z_exp_real = Z_exp[:,1]

# Z imaginary component
Z_exp_imag = Z_exp[:,2]




#______________________________________________________________________________
# 2) ...........       Define RBF and TCDS parameters 
#______________________________________________________________________________

tau_plot  = 1/freq_vec

# Define Gaussian basis function parameters
gaussian_coeff = 0.1
width_coeff = 1.6651
D_f = np.mean(np.diff(np.log(1./freq_vec)))
epsilon  = gaussian_coeff*width_coeff/D_f
L=0

# Define TCDS parameters
#time-constant-domain range and time-constant-domain vector
taumax=ceil(max(np.log10(1./(freq_vec))))+1    
taumin=floor(min(np.log10(1./(freq_vec))))-1
freq_out = np.logspace(-taumin, -taumax, 10*(len(freq_vec)))
tau_out = 1./ freq_out

#______________________________________________________________________________
# 3) ...........  Build regression matrices for regularization
#______________________________________________________________________________

TCDS_Z_im = Z_primeprime(tau_plot,freq_vec, epsilon, L)
TCDS_Z_re = Z_prime(tau_plot,freq_vec, epsilon, L)

#______________________________________________________________________________
# 4) ...........  Optimization problem
#               Cost function: f(β)=∥Xβ−Y∥22+λ∥β∥22,
#    β: parameters    Y: Z_exp     X: TCDS_Z    lambda: hyperparameter
#______________________________________________________________________________

n = len(freq_vec) + 2
beta = cp.Variable(n)                #parameters to be estimated
hyper = cp.Parameter(nonneg=True)    #lambda to be tunned
problem =  cp.Problem(cp.Minimize(objective_fun(TCDS_Z_im, -Z_exp_imag, beta, hyper)),
                     [beta >= 0])

train_errors = []
beta_values = []
hyper.value = 1E-2                  #lambda to be tunned
problem.solve()
train_errors.append(mse(TCDS_Z_im, -Z_exp_imag, beta))
beta_values.append(beta.value)

# Final estimated parameters
theta = np.transpose(np.array(beta_values,dtype=float))
theta_hat = theta[2:len(freq_vec)+2]

 
#______________________________________________________________________________
# 5) ...........        Final TCDS computation
#               Interpolation usage for resolution enhacement
#______________________________________________________________________________ 
    
TCDS_interp = interpol_TCDS(freq_out,freq_vec,theta_hat,epsilon)
 

#______________________________________________________________________________
# 5) ...........        Plot final results
#               a) Impedance Spectrum (Nyquist plot)
#               b) Time-constant-domain Spectrum
#______________________________________________________________________________ 

# Nyquist plot of the impedance data
plt.figure()
plt.plot(Z_exp_real, -Z_exp_imag, "o", markersize=10, color="red")
plt.xlabel('Real Z')
plt.ylabel('-Imag Z')

# TCDS plot of the distribution function
plt.figure()
plt.semilogx(tau_out,TCDS_interp)
plt.xlabel('tau (s)')
plt.ylabel('gamma ')
plt.show()
