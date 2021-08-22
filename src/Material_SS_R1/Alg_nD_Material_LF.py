#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:19:08 2020

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
import matplotlib.pyplot as plt
import pickle

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()
from LimitStateFunctions import LimitStateFunctions as LSF
from ML_TF import ML_TF
from DrawRandom import DrawRandom as DR
from pyDOE import *

Ndim = 8
Ndim0 = 5
value = 400.0

LS1 = LSF()
DR1 = DR()
num_s = 500

# np.random.seed(1011)
# sto = np.zeros((8,10000))
# for ii in np.arange(0,10000,1):
#     inp, inp1 = (DR1.MaterialRandom())
#     sto[:,ii] = inp


## Monte Carlo simulations

# Nsims = int(10)
# y = np.zeros(Nsims)4
# ys = np.zeros(Nsims)
# LS1 = LSF()
# DR1 = DR()
# Ndim = 8
# value = 0.14

# for ii in np.arange(0,Nsims,1):
#     inp, inp1 = (DR1.MaterialRandom())
#     inpp = inp[None,:]
#     y[ii] = np.array(LS1.Material_HF(inpp))
#     inpp = inp1[None,:]
#     ys[ii] = np.array(LS1.Material_LF(inpp))
#     print(ii/Nsims)

# req = len(np.rot90(np.where(y>value)))/Nsims

# req = 0.0001469 (2.4e6 simulations)

# Basic subset simulation

LS1 = LSF()
DR1 = DR()
num_s = 500

uni = uniform()
Nsub = 15000
Psub = 0.1
Nlim = 4
y1 = np.zeros((Nsub,Nlim))
y1_lim = np.zeros(Nlim)
y1_lim[Nlim-1] = value
inp1 = np.zeros((Nsub,Ndim,Nlim))
rv = norm(loc=0,scale=1)
y_seed = np.zeros(int(Psub*Nsub))
Indicator = np.ones((Nsub,Nlim))

for ii in np.arange(0,Nsub,1):
    inp, inp_LF = (DR1.MaterialRandom())
    inpp = inp[None,:]
    y1[ii,0] = np.array(LS1.Material_HF(inpp))
    inp1[ii,:,0] = inp
    print(ii)

inpp = np.zeros(Ndim)
count_max = Nsub/(Psub*Nsub)
seeds_outs = np.zeros(int(Psub*Nsub))
seeds = np.zeros((int(Psub*Nsub),Ndim))
markov_seed = np.zeros(Ndim)
markov_out = 0.0
r_sto = np.zeros((Nsub-int(Psub*Nsub),Nlim-1,Ndim))
# ind_sto = 3
prop_std_req = np.array([0.1,0.1,0.1,0.1,0.1,0.5,0.5,0.5])*0.75
std_prop = np.zeros(Ndim)

for kk in np.arange(1,2,1):
    count = np.inf
    ind_max = 0
    ind_sto = -1
    seeds_outs = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
    y1_lim[kk-1] = np.min(seeds_outs)
    k = (y1[:,kk-1]).argsort()
    indices = k[int((1-Psub)*Nsub):(len(y1))]
    seeds = inp1[indices,:,kk-1]
    std_prop = np.log((seeds)).std(0)

    for ii in np.arange(5649,Nsub,1):
        print(ii)
        print(kk)
        nxt = np.zeros((1,Ndim))
        if count > count_max:
            ind_sto = ind_sto + 1
            count = 0
            markov_seed = seeds[ind_sto,:]
            markov_out = seeds_outs[ind_sto]
        else:
            markov_seed = inp1[ii-1,:,kk]
            markov_out = y1[ii-1,kk]

        count = count + 1

        for jj in np.arange(0,Ndim,1):
            rv1 = norm(loc=np.log(markov_seed[jj]),scale=std_prop[jj])
            prop = np.exp(rv1.rvs())
            # rv1 = norm(loc=np.log(inp1[ind_max,jj,kk]),scale=prop_std_req[jj])
            # prop = np.exp(rv1.rvs())
            # rv1 = uniform(loc=(np.log(inp1[ind_max,jj,kk])-prop_std_req[jj]),scale=(2*prop_std_req[jj]))
            # prop = np.exp(rv1.rvs())
            r = np.log(DR1.MaterialPDF(rv_req=prop, index=jj, LF=0)) - np.log(DR1.MaterialPDF(rv_req=(markov_seed[jj]),index=jj,LF=0)) # rv.pdf((prop))/rv.pdf((inp1[ii-(int(Psub*Nsub)),jj,kk]))
            r_sto[ii-(int(Psub*Nsub)),kk-1,jj] = r
            if r>np.log(uni.rvs()):
                nxt[0,jj] = prop
            else:
                nxt[0,jj] = markov_seed[jj]
            inpp[jj] = nxt[0,jj]
        y_nxt = np.array(LS1.Material_HF(inpp[None,:])).reshape(1)
        if y_nxt>y1_lim[kk-1]:
            inp1[ii,:,kk] = inpp # np.array([nxt[0,0], nxt[0,1], nxt[0,2]])
            y1[ii,kk] = y_nxt
        else:
            inp1[ii,:,kk] = markov_seed
            y1[ii,kk] = markov_out
            Indicator[ii,kk] = 0.0

Pf = 1
Pi_sto = np.zeros(Nlim)
cov_sq = 0
for kk in np.arange(0,Nlim,1):
    Pi = len(np.rot90(np.where(y1[:,kk]>np.min([y1_lim[kk],value]))))/(Nsub)
    Pf = Pf * Pi
    Pi_sto[kk] = Pi
    cov_sq = cov_sq + ((1-Pi)/(Pi*Nsub))
cov_req = np.sqrt(cov_sq)
