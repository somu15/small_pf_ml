#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 10:53:37 2021

@author: dhulls
"""


from os import sys
import os
import pathlib
import numpy as np
import random
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import rayleigh
from scipy.stats import uniform
from scipy.stats import cauchy
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()
from pyDOE import *

mpl.rcParams['axes.linewidth'] = 1.5
plt.rc('font', family='serif', size=20)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rcParams.update({
    "text.usetex": True})

## Standard Subset Simulation

n_subs = 4
cov_i = np.zeros(n_subs)
for ss in np.arange(0,n_subs,1):
    sub_req = ss
    rho_req = np.zeros(9)
    stos = np.zeros(Nsub)
    for kk in np.arange(0,10,1):
        sum_req = 0.0
        for ii in np.arange(0,int(Nsub/10),1):
            for jj in np.arange(0,10-kk,1):
                ind1 = (10*ii-1)+jj
                ind2 = (10*ii-1)+jj+kk
                stos[ind1] = int(y1[ind1,sub_req]>y1_lim[sub_req])
                # print(ind1)
                sum_req = sum_req + int(y1[ind1,sub_req]>y1_lim[sub_req]) * int(y1[ind2,sub_req]>y1_lim[sub_req])
        rho_req[kk-1] = 1/(Pi_sto[sub_req]*(1-Pi_sto[sub_req])) * ((1/(int(Nsub/10)*(10-kk))) * sum_req - Pi_sto[sub_req]**2)
    
    gamma = 0.0
    for ii in np.arange(0,9,1):
        gamma = gamma + 2 * (1-(ii+1)/10) * rho_req[ii]
    
    cov_i[ss] = ((1-Pi_sto[sub_req])/(Pi_sto[sub_req] * Nsub)*(1+gamma))

cov_actual = np.sqrt(np.sum(cov_i))

# n_subs = 4
# cov_i = np.zeros(n_subs)
# for ss in np.arange(0,n_subs,1):
#     sub_req = ss
#     rho_req = np.zeros(9)
#     stos = np.zeros(Nsub)
#     for kk in np.arange(0,10,1):
#         sum_req = 0.0
#         for ii in np.arange(0,int(Nsub/10),1):
#             for jj in np.arange(0,10-kk,1):
#                 ind1 = (10*ii-1)+jj
#                 ind2 = (10*ii-1)+jj+kk
#                 stos[ind1] = int(y2[ind1,sub_req]>y1_lim[sub_req])
#                 # print(ind1)
#                 sum_req = sum_req + int(y2[ind1,sub_req]>y1_lim[sub_req]) * int(y2[ind2,sub_req]>y1_lim[sub_req])
#         rho_req[kk-1] = 1/(Pi_sto[sub_req]*(1-Pi_sto[sub_req])) * ((1/(int(Nsub/10)*(10-kk))) * sum_req - Pi_sto[sub_req]**2)
    
#     gamma = 0.0
#     for ii in np.arange(0,9,1):
#         gamma = gamma + 2 * (1-(ii+1)/10) * rho_req[ii]
    
#     cov_i[ss] = ((1-Pi_sto[sub_req])/(Pi_sto[sub_req] * Nsub)*(1+gamma))

# cov_actual = np.sqrt(np.sum(cov_i))

## Proposed Algorithm 1

# sub_req = 1
# rho_req_n = np.zeros(9)

# for kk in np.arange(0,10,1):
#     # print(kk)
#     sum_req = 0.0
#     for ii in np.arange(0,int(Nsub/10),1):
#         for jj in np.arange(0,10-kk,1):
#             ind1 = (10*ii-1)+jj
#             ind2 = (10*ii-1)+jj+kk
            
#             fact1 = 0.977
#             fact2 = 0.977
            
#             if int(y1[ind1,sub_req]>y1_lim[sub_req]) == 1:
#                 tmp1 = fact1
#             else:
#                 tmp1 = 1-fact1
                
#             sum1 = sum1 + tmp1
            
                
#             if int(y1[ind2,sub_req]>y1_lim[sub_req]) == 1:
#                 tmp2 = fact2
#             else:
#                 tmp2 = 1-fact2
            
#             prod1 = tmp1
#             prod2 = tmp2
            
#             # prod1 = int(y1[ind1,sub_req]>y1_lim[sub_req])
#             # prod2 = int(y1[ind2,sub_req]>y1_lim[sub_req])
            
#             sum_req = sum_req + prod1 * prod2
    
#     rho_req_n[kk-1] = 1/(Pi_sto[sub_req]*(1-Pi_sto[sub_req])) * ((1/(int(Nsub/10)*(10-kk))) * sum_req - Pi_sto[sub_req]**2)

# gamma_n = 0.0
# for ii in np.arange(0,9,1):
#     gamma_n = gamma_n + 2 * (1-(ii+1)/10) * rho_req_n[ii]

# cov_in = np.sqrt((1-Pi_sto[sub_req])/(Pi_sto[sub_req] * Nsub)*(1+gamma_n))

## Proposed Algorithm 2

# sub_req = 1
# rho_req_n = np.zeros(9)


# for kk in np.arange(0,10,1):
#     # print(kk)
#     sum_req = 0.0
#     sum_var = 0
#     for ii in np.arange(0,int(Nsub/10),1):
#         for jj in np.arange(0,10,1):
#             ind1 = (10*ii-1)+jj
#             fact1 = 1
#             if int(y1[ind1,sub_req]>y1_lim[sub_req]) == 1:
#                 tmp1 = fact1
#             else:
#                 tmp1 = 0
#             sum_var = sum_var + tmp1*tmp1
#     sum_var = sum_var/Nsub-Pi_sto[sub_req]**2
            
#     for ii in np.arange(0,int(Nsub/10),1):
#         for jj in np.arange(0,10-kk,1):
#             ind1 = (10*ii-1)+jj
#             ind2 = (10*ii-1)+jj+kk
            
#             fact1 = 0.9
#             fact2 = 0.9
            
#             if int(y1[ind1,sub_req]>y1_lim[sub_req]) == 1:
#                 tmp1 = fact1
#             else:
#                 tmp1 = 0 # 1-fact1
                
#             sum1 = sum1 + tmp1
            
                
#             if int(y1[ind2,sub_req]>y1_lim[sub_req]) == 1:
#                 tmp2 = fact2
#             else:
#                 tmp2 = 0 # 1-fact2
            
#             prod1 = tmp1
#             prod2 = tmp2
            
#             # prod1 = int(y1[ind1,sub_req]>y1_lim[sub_req])
#             # prod2 = int(y1[ind2,sub_req]>y1_lim[sub_req])
            
#             sum_req = sum_req + prod1 * prod2 # - prod1*Pi_sto[sub_req] - prod2*Pi_sto[sub_req]
#     #  (Pi_sto[sub_req]*(1-Pi_sto[sub_req]))
#     rho_req_n[kk-1] = 1/sum_var * ((1/(int(Nsub/10)*(10-kk))) * sum_req - Pi_sto[sub_req]**2)

# gamma_n = 0.0
# for ii in np.arange(0,9,1):
#     gamma_n = gamma_n + 2 * (1-(ii+1)/10) * rho_req_n[ii]

# cov_in = np.sqrt((1-Pi_sto[sub_req])/(Pi_sto[sub_req] * Nsub)*(1+gamma_n))


## Proposed Algorithm FINAL

# sub_req = 2
# rho_req_n = np.zeros(9)

# for kk in np.arange(0,10,1):
#     print(kk)
#     sum_req = 0.0
#     sum_var = 0
#     for ii in np.arange(0,int(Nsub/10),1):
#         for jj in np.arange(0,10,1):
#             ind1 = (10*ii-1)+jj
#             fact1 = norm().cdf(u_GP[ind1,sub_req])
#             fact3 = norm().cdf(u_req[ind1])
#             if int(y1[ind1,sub_req]>y1_lim[sub_req]) == 1:
#                 tmp1 = fact1*fact3
#             else:
#                 tmp1 = fact1*(1-fact3)
#             sum_var = sum_var + tmp1*tmp1
#     sum_var = sum_var/Nsub-Pi_sto[sub_req]**2
            
#     for ii in np.arange(0,int(Nsub/10),1):
#         for jj in np.arange(0,10-kk,1):
#             ind1 = (10*ii-1)+jj
#             ind2 = (10*ii-1)+jj+kk
            
#             fact1 = norm().cdf(u_GP[ind1,sub_req])
#             fact2 = norm().cdf(u_GP[ind2,sub_req])
#             fact3 = norm().cdf(u_req[ind1])
#             fact4 = norm().cdf(u_req[ind2])
            
#             if int(y1[ind1,sub_req]>y1_lim[sub_req]) == 1:
#                 tmp1 = fact1*fact3
#             else:
#                 tmp1 = fact1*(1-fact3)
                
#             # sum1 = sum1 + tmp1
            
                
#             if int(y1[ind2,sub_req]>y1_lim[sub_req]) == 1:
#                 tmp2 = fact2*fact4
#             else:
#                 tmp2 = fact2*(1-fact4)
            
#             prod1 = tmp1
#             prod2 = tmp2
            
#             # prod1 = int(y1[ind1,sub_req]>y1_lim[sub_req])
#             # prod2 = int(y1[ind2,sub_req]>y1_lim[sub_req])
            
#             sum_req = sum_req + prod1 * prod2 # - prod1*Pi_sto[sub_req] - prod2*Pi_sto[sub_req]
#     #  (Pi_sto[sub_req]*(1-Pi_sto[sub_req]))
#     rho_req_n[kk-1] = 1/sum_var * ((1/(int(Nsub/10)*(10-kk))) * sum_req - Pi_sto[sub_req]**2)

# gamma_n = 0.0
# for ii in np.arange(0,9,1):
#     gamma_n = gamma_n + 2 * (1-(ii+1)/10) * rho_req_n[ii]


sub_req = 2
rho_req_n = np.zeros(9)

for kk in np.arange(0,10,1):
    print(kk)
    sum_req = 0.0
    sum_var = 0
    for ii in np.arange(0,int(Nsub/10),1):
        for jj in np.arange(0,10,1):
            ind1 = (10*ii-1)+jj
            fact1 = norm().cdf(u_GP[ind1,sub_req])
            # fact3 = norm().cdf(u_req[ind1])
            if int(y1[ind1,sub_req]>y1_lim[sub_req]) == 1:
                tmp1 = fact1
            else:
                tmp1 = 1-fact1
            sum_var = sum_var + tmp1*tmp1
    sum_var = sum_var/Nsub-Pi_sto[sub_req]**2
            
    for ii in np.arange(0,int(Nsub/10),1):
        for jj in np.arange(0,10-kk,1):
            ind1 = (10*ii-1)+jj
            ind2 = (10*ii-1)+jj+kk
            
            fact1 = norm().cdf(u_GP[ind1,sub_req])
            fact2 = norm().cdf(u_GP[ind2,sub_req])
            
            if int(y1[ind1,sub_req]>y1_lim[sub_req]) == 1:
                tmp1 = fact1
            else:
                tmp1 = 1-fact1
                
            # sum1 = sum1 + tmp1
            
                
            if int(y1[ind2,sub_req]>y1_lim[sub_req]) == 1:
                tmp2 = fact2
            else:
                tmp2 = 1-fact2
            
            prod1 = tmp1
            prod2 = tmp2
            
            # prod1 = int(y1[ind1,sub_req]>y1_lim[sub_req])
            # prod2 = int(y1[ind2,sub_req]>y1_lim[sub_req])
            
            sum_req = sum_req + prod1 * prod2 # - prod1*Pi_sto[sub_req] - prod2*Pi_sto[sub_req]
    #  (Pi_sto[sub_req]*(1-Pi_sto[sub_req]))
    rho_req_n[kk-1] = 1/sum_var * ((1/(int(Nsub/10)*(10-kk))) * sum_req - Pi_sto[sub_req]**2)

gamma_n = 0.0
for ii in np.arange(0,9,1):
    gamma_n = gamma_n + 2 * (1-(ii+1)/10) * rho_req_n[ii]

# cov_in = np.sqrt((1-Pi_sto[sub_req])/(Pi_sto[sub_req] * Nsub)*(1+gamma_n))

# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(np.arange(0,9,1),rho_req, c='tab:blue', label='Subset simulation')
# ax.scatter(np.arange(0,9,1),rho_req_n, s=50, c='tab:orange', label='Proposed algorithm')
# ax.set_xlabel('Lag length')
# ax.set_ylabel('Autocorrelation')
# plt.legend(frameon=False)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(1, 1, 1)
ax.hist(u_GP[:,1],range=[2, 20.0])
ax.set_xlabel('Count')
ax.set_ylabel('U function value')
