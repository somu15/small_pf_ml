#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:35:52 2021

@author: dhulls
"""

from os import sys
import pathlib
import numpy as np
import random
from scipy.stats import norm
from scipy.stats import uniform
import matplotlib.pyplot as plt

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
value = 270.0

## Monte Carlo

LS1 = LSF()
DR1 = DR()
num_s = 500
uni = uniform()
Nsub = int(17e6)
y1 = np.zeros(Nsub)
inp1 = np.zeros((Nsub,8))

for ii in np.arange(0,Nsub,1):
    inp = DR1.BoreholeRandom().reshape(Ndim)
    inpp = inp[None,:]
    print(ii/Nsub)
    y1[ii] = np.array((LS1.Scalar_Borehole_HF_nD(inpp))).reshape(1)
    inp1[ii,:] = inp



def Convert(lst):
    return [ -i for i in lst ]

## Subset simultion with HF-LF and GP

LS1 = LSF()
DR1 = DR()
num_s = 500
uni = uniform()
Nsub = 40000
Psub = 0.1
Nlim = 5
y1 = np.zeros((Nsub,Nlim))
y1_lim = np.zeros(Nlim)
y1_lim[Nlim-1] = value
inp1 = np.zeros((Nsub,Ndim,Nlim))
rv = norm(loc=0,scale=1)
u_lim_vec = np.array([2,2,2,2,2,2,2,2,2])
Indicator = np.ones((Nsub,Nlim))

for ii in np.arange(0,Nsub,1):
    inp = DR1.BoreholeRandom().reshape(Ndim)
    inpp = inp[None,:]
    print(ii)
    y1[ii,0] = np.array((LS1.Scalar_Borehole_HF_nD(inpp))).reshape(1)
    inp1[ii,:,0] = inp

inpp = np.zeros((1,Ndim))
count_max = int(Nsub/(Psub*Nsub))-1
seeds_outs = np.zeros(int(Psub*Nsub))
seeds = np.zeros((int(Psub*Nsub),Ndim))
markov_seed = np.zeros(Ndim)
markov_out = 0.0
std_prop = np.zeros(Ndim)

prop_std_req = np.array([0.0216,0.75,11373.07,25.98,11.453,25.98,121.243,474.148])

for kk in np.arange(1,Nlim,1):
    count = np.inf
    ind_max = 0
    ind_sto = -1
    seeds_outs = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
    y1_lim[kk-1] = np.min(seeds_outs)
    k = (y1[:,kk-1]).argsort()
    indices = k[int((1-Psub)*Nsub):(len(y1))]
    seeds = inp1[indices,:,kk-1]
    std_prop = np.log((seeds)).std(0)
    tmp = np.zeros((len(seeds_outs),1+Ndim))
    tmp[:,0] = seeds_outs
    tmp[:,1:9] = seeds
    np.random.shuffle(tmp)
    seeds_outs = tmp[:,0]
    seeds = tmp[:,1:9]

    for ii in np.arange(0,(Nsub),1):
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
            if jj == 0:
                rv1 = norm(loc=np.log(markov_seed[jj]),scale=std_prop[jj]) # 0.1
            else:
                rv1 = norm(loc=np.log(markov_seed[jj]),scale=std_prop[jj]) # 1.0
            # rv1 = norm(loc=np.log(inp1[ind_max,jj,kk]),scale=0.5)
            prop = np.exp(rv1.rvs())
            # if jj == 1:
            #     rv1 = rv1 = uniform(loc=(np.log(markov_seed[jj])-prop_std_req[jj]),scale=(2*prop_std_req[jj]))
            #     prop = np.exp(rv1.rvs())
            # else:
            #     rv1 = uniform(loc=((markov_seed[jj])-prop_std_req[jj]),scale=(2*prop_std_req[jj]))
            #     prop = (rv1.rvs())



            r = np.log(DR1.BoreholePDF(rv_req=prop, index=jj)) - np.log(DR1.BoreholePDF(rv_req=(markov_seed[jj]),index=jj))
            if r>np.log(uni.rvs()):
                nxt[0,jj] = prop
            else:
                nxt[0,jj] = markov_seed[jj]
            inpp[0,jj] = nxt[0,jj]
        y_nxt = np.array((LS1.Scalar_Borehole_HF_nD(inpp))).reshape(1)
        print(ii)
        print(kk)
        # print(y_nxt)
        # print(y1_lim[kk-1])
        if (y_nxt)>y1_lim[kk-1]:
            inp1[ii,:,kk] = inpp
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
