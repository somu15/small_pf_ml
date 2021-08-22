#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 15:54:39 2021

@author: dhulls
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from scipy.stats import uniform
from distutils.util import strtobool
np.random.seed(0)

def WeibullF(s,s0,m):
    return 1-np.exp(-(s/s0)**m)

def WeibullInv(F,s0,m):
    return (s0 * np.power(-np.log(1-F),1/m)) 

# sx = np.arange(0,1000,1)
s0 = 800
m0 = 3.0

####### Varied Weibull parameters ##########

Ntot = 200000
F_est0 = np.zeros(Ntot)
F_est1 = np.zeros(Ntot)
F_est2 = np.zeros(Ntot)
for ii in np.arange(0,Ntot,1):
    S = np.exp(norm(np.log(5),1.0).rvs(1))
    # s = np.exp(norm(loc = np.log(s0), scale = 0.2).rvs())
    # m = np.exp(norm(loc = np.log(m0), scale = 0.2).rvs())
    F_est0[ii] = WeibullF(S,s0,m0)
    # s1 = np.exp(norm(loc = np.log(s0), scale = 0.1).rvs())
    # m1 = np.exp(norm(loc = np.log(m0), scale = 0.1).rvs())
    # F_est1[ii] = WeibullF(S,s1,m1)
    # s2 = np.exp(norm(loc = np.log(s0), scale = 0.2).rvs())
    # m2 = np.exp(norm(loc = np.exp(m0), scale = 0.1).rvs())
    # F_est2[ii] = WeibullF(S,s2,m2)


N_moving = np.arange(1,Ntot+1,1)

Pf_moving1 = np.cumsum(F_est0)/N_moving
tmp = np.cumsum(F_est0**2)/N_moving - Pf_moving1**2
COV_moving1 = np.sqrt(tmp/N_moving)/Pf_moving1

# Pf_moving1 = np.cumsum(F_est1)/N_moving
# tmp = np.cumsum(F_est1**2)/N_moving - Pf_moving1**2
# COV_moving1 = np.sqrt(tmp/N_moving)/Pf_moving1

# Pf_moving2 = np.cumsum(F_est2)/N_moving
# tmp = np.cumsum(F_est2**2)/N_moving - Pf_moving2**2
# COV_moving2 = np.sqrt(tmp/N_moving)/Pf_moving2

####### Fixed Weibull parameters ###########

# F = WeibullF(sx,s0,m)

# S_req1 = np.exp(norm(np.log(5),1).rvs(1000000))
# F_ans1 = WeibullF(S_req1,s0,m)
# F_moving1 = np.cumsum(F_ans1)/np.arange(1,1000001,1)
# S_req2 = np.exp(norm(np.log(5),1).rvs(1000000))
# F_ans2 = WeibullF(S_req2,s0,m)
# F_moving2 = np.cumsum(F_ans2)/np.arange(1,1000001,1)
# S_req3 = np.exp(norm(np.log(5),1).rvs(1000000))
# F_ans3 = WeibullF(S_req3,s0,m)
# F_moving3 = np.cumsum(F_ans3)/np.arange(1,1000001,1)
# S_req4 = np.exp(norm(np.log(5),1).rvs(1000000))
# F_ans4 = WeibullF(S_req4,s0,m)
# F_moving4 = np.cumsum(F_ans4)/np.arange(1,1000001,1)
# S_req5 = np.exp(norm(np.log(5),1).rvs(1000000))
# F_ans5 = WeibullF(S_req5,s0,m)
# F_moving5 = np.cumsum(F_ans5)/np.arange(1,1000001,1)
# cov_F = np.std(np.array([np.mean(F_ans1),np.mean(F_ans2),np.mean(F_ans3),np.mean(F_ans4),np.mean(F_ans5)]))/np.mean(np.array([np.mean(F_ans1),np.mean(F_ans2),np.mean(F_ans3),np.mean(F_ans4),np.mean(F_ans5)]))
# cov_movF = np.std(np.array([(F_moving1),(F_moving2),(F_moving3),(F_moving4),(F_moving5)]),axis=0)/np.mean(np.array([(F_moving1),(F_moving2),(F_moving3),(F_moving4),(F_moving5)]),axis=0)

# U_rand = uniform().rvs(1000000)
# Strens = WeibullInv(U_rand,s0,m)
# I_ans1 = np.multiply(S_req1>Strens,1)
# I_moving1 = np.cumsum(I_ans1)/np.arange(1,1000001,1)
# U_rand = uniform().rvs(1000000)
# Strens = WeibullInv(U_rand,s0,m)
# I_ans2 = np.multiply(S_req2>Strens,1)
# I_moving2 = np.cumsum(I_ans2)/np.arange(1,1000001,1)
# U_rand = uniform().rvs(1000000)
# Strens = WeibullInv(U_rand,s0,m)
# I_ans3 = np.multiply(S_req3>Strens,1)
# I_moving3 = np.cumsum(I_ans3)/np.arange(1,1000001,1)
# U_rand = uniform().rvs(1000000)
# Strens = WeibullInv(U_rand,s0,m)
# I_ans4 = np.multiply(S_req4>Strens,1)
# I_moving4 = np.cumsum(I_ans4)/np.arange(1,1000001,1)
# U_rand = uniform().rvs(1000000)
# Strens = WeibullInv(U_rand,s0,m)
# I_ans5 = np.multiply(S_req5>Strens,1)
# I_moving5 = np.cumsum(I_ans5)/np.arange(1,1000001,1)
# cov_I = np.std(np.array([np.mean(I_ans1),np.mean(I_ans2),np.mean(I_ans3),np.mean(I_ans4),np.mean(I_ans5)]))/np.mean(np.array([np.mean(I_ans1),np.mean(I_ans2),np.mean(I_ans3),np.mean(I_ans4),np.mean(I_ans5)]))
# cov_movI = np.std(np.array([(I_moving1),(I_moving2),(I_moving3),(I_moving4),(I_moving5)]),axis=0)/np.mean(np.array([(I_moving1),(I_moving2),(I_moving3),(I_moving4),(I_moving5)]),axis=0)

# plt.plot(np.arange(1,1000001,1),F_moving1,label='Weibull failure')
# plt.plot(np.arange(1,1000001,1),I_moving1,label='Indicator function')
# plt.xlabel('Sample number')
# plt.ylabel('Failure probability')
# plt.legend()

# plt.semilogy(np.arange(1,1000001,1),cov_movF,label='Weibull failure')
# plt.plot(np.arange(1,1000001,1),cov_movI,label='Indicator function')
# plt.xlabel('Sample number')
# plt.ylabel('Coefficient of Variation')
# plt.legend()
