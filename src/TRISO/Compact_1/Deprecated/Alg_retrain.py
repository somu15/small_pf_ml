
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 10:09:55 2021

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
value = 0.0

LS1 = LSF()
DR1 = DR()
num_s = 500
P = np.array([2.7282643229905267e-05,5.230937638444862e-05,1.5642320384140967e-05,7.42526553558767e-06,2.0917204816879883e-05,1,1,1])

## Training GP

def Norm1(X1,X,dim):
    K = np.zeros((len(X1),dim))
    for ii in np.arange(0,dim,1):
        K[:,ii] = X1[:,ii]/X[ii]
    return K

def Norm3(X1,X):
    return (X1/1.2e9)

def InvNorm3(X1,X):
    return (X1)

## Train the LF model

Ninit_GP = 40
lhd = DR1.TrisoLHS(Ninit_GP) # DR1.TrisoRandom(N=Ninit_GP) #  uniform(loc=-3.5,scale=7.0).ppf(lhd0) #
inp_LFtrain = lhd
y_HF_LFtrain = Norm3(LS1.Triso_1d(inp_LFtrain),1)
ML0 = ML_TF(obs_ind = Norm1(inp_LFtrain,P,Ndim), obs = (y_HF_LFtrain))
amp0, len0 = ML0.GP_train(amp_init=1., len_init=1., num_iters = 1000)

## Train the GP diff model

Ninit_GP = 12
lhd = DR1.TrisoRandom(N=Ninit_GP)
inp_GPtrain = lhd
samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, pred_ind = Norm1(inp_GPtrain,P,Ndim), num_samples=num_s)
y_LF_GP = np.array(InvNorm3(np.mean(np.array(samples0),axis=0),y_HF_LFtrain))
y_HF_GP = Norm3(np.array((LS1.Triso_1d(inp_GPtrain))),1)
y_GPtrain = y_HF_GP - y_LF_GP
ML = ML_TF(obs_ind = Norm1(inp_GPtrain,P,Ndim), obs = (y_GPtrain))
amp1, len1 = ML.GP_train(amp_init=1., len_init=1., num_iters = 1000)
Iters = 400
amp_sto = np.array(amp1).reshape(1)
len_sto = np.array(len1).reshape(1)

## Subset simultion with HF-LF and GP

uni = uniform()
Nsub = 2500 # 10000
Psub = 0.1
Nlim = 4
y1 = np.zeros((Nsub,Nlim))
ylf = np.zeros((Nsub,Nlim))
y1_lim = np.zeros(Nlim)
y1_lim[Nlim-1] = value
inp1 = np.zeros((Nsub,Ndim,Nlim))
rv = norm(loc=0,scale=1)
u_lim_vec = np.array([2,2,2,2,2,2,2,2,2])

u_GP = np.zeros((Nsub,Nlim))
subs_info = np.zeros((Nsub,Nlim))
LF_plus_GP = np.empty(1, dtype = float)
GP_pred = np.empty(1, dtype = float)
additive = value
Indicator = np.ones((Nsub,Nlim))
counter = 1

for ii in np.arange(0,Nsub,1):
    inp = DR1.TrisoRandom()
    inp = inp.reshape(Ndim)
    inpp = inp[None,:]
    samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, pred_ind = Norm1(inpp,P,Ndim), num_samples=num_s)
    LF = np.array(InvNorm3(np.mean(np.array(samples0),axis=0),y_HF_LFtrain))
    inp1[ii,:,0] = inp
    if ii <Ninit_GP:
        additive = np.percentile(y_HF_GP,90)
    else:
        additive = np.percentile(y1[0:ii,0],90)
    samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inpp,P,Ndim), num_samples=num_s)
    GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
    u_check = (np.abs(LF + GP_diff - additive))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
    u_GP[ii,0] = u_check

    u_lim = 2.0
    print(ii)
    if u_check > u_lim:
        y1[ii,0] = LF + GP_diff
    else:
        y1[ii,0] = (np.array(Norm3(LS1.Triso_1d(inpp),1)).reshape(1))
        inp_GPtrain = np.concatenate((inp_GPtrain, inpp.reshape(1,Ndim)))

        inp_LFtrain = np.concatenate((inp_LFtrain, inpp.reshape(1,Ndim)))
        y_HF_LFtrain = np.concatenate((y_HF_LFtrain, (y1[ii,0].reshape(1))))
        ML0 = ML_TF(obs_ind = Norm1(inp_LFtrain,P,Ndim), obs = (y_HF_LFtrain))
        amp0, len0 = ML0.GP_train(amp_init=1., len_init=1., num_iters = Iters)
        samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, pred_ind = Norm1(inp_GPtrain,P,Ndim), num_samples=num_s)
        y_LF_GP = np.array(InvNorm3(np.mean(np.array(samples0),axis=0),y_HF_LFtrain))
        y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
        y_GPtrain = y_HF_GP - y_LF_GP

        ML = ML_TF(obs_ind = Norm1(inp_GPtrain,P,Ndim), obs = (y_GPtrain))
        amp1, len1 = ML.GP_train(amp_init=1., len_init=1., num_iters = Iters)
        amp_sto = np.concatenate((amp_sto, np.array(amp1).reshape(1)))
        len_sto = np.concatenate((len_sto, np.array(len1).reshape(1)))
        subs_info[ii,0] = 1.0

inpp = np.zeros((1,Ndim))
count_max = int(Nsub/(Psub*Nsub))-1
seeds_outs = np.zeros(int(Psub*Nsub))
seeds = np.zeros((int(Psub*Nsub),Ndim))
markov_seed = np.zeros(Ndim)
markov_out = 0.0
u_req = np.zeros(Nsub)
u_check1 = 10.0
std_prop = np.zeros(Ndim)
LF_prev = 0.0

for kk in np.arange(1,Nlim,1):
    count = np.inf
    ind_max = 0
    ind_sto = -1
    seeds_outs = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
    y1_lim[kk-1] = np.min(seeds_outs)
    k = (y1[:,kk-1]).argsort()
    indices = k[int((1-Psub)*Nsub):(len(y1))]
    seeds = inp1[indices,:,kk-1]
    std_prop = np.log(np.abs(seeds)).std(0)
    tmp = np.zeros((len(seeds_outs),1+Ndim))
    tmp[:,0] = seeds_outs
    tmp[:,1:9] = seeds
    np.random.shuffle(tmp)
    seeds_outs = tmp[:,0]
    seeds = tmp[:,1:9]

    for ii in np.arange(0,(Nsub),1):
        print(kk)
        print(ii)
        nxt = np.zeros((1,Ndim))

        if count > count_max:
            ind_sto = ind_sto + 1
            count = 0
            markov_seed = seeds[ind_sto,:]
            markov_out = seeds_outs[ind_sto]
        # else:
        #     markov_seed = inp1[ii-1,:,kk]
        #     markov_out = y1[ii-1,kk]

        count = count + 1

        for jj in np.arange(0,Ndim,1):
            rv1 = norm(loc=np.log(markov_seed[jj]),scale=std_prop[jj]) # 0.75
            prop = np.exp(rv1.rvs())
            r = np.log(DR1.TrisoPDF(rv_req=prop, index=jj)) - np.log(DR1.TrisoPDF(rv_req=(markov_seed[jj]),index=jj))
            if r>np.log(uni.rvs()):
                nxt[0,jj] = prop
            else:
                nxt[0,jj] = markov_seed[jj]
            inpp[0,jj] = nxt[0,jj]

        samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, pred_ind = Norm1(inpp,P,Ndim), num_samples=num_s)
        LF = np.array(InvNorm3(np.mean(np.array(samples0),axis=0),y_HF_LFtrain))
        ylf[ii,kk] = LF
        samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inpp,P,Ndim), num_samples=num_s)
        GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
        if kk<(Nlim-1):
            if ii < Ninit_GP:
                additive =  y1_lim[kk-1]
            else:

                additive = np.percentile(y1[0:ii,kk],90)
        else:
            additive = value

        u_check = (np.abs(LF + GP_diff - additive))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)

        u_GP[ii,kk] = u_check
        u_lim = u_lim_vec[kk]

        if u_check > u_lim and ii>=Ninit_GP:
            y_nxt = LF + GP_diff
        else:
            y_nxt = (np.array(Norm3(LS1.Triso_1d(inpp),1)).reshape(1))
            inp_GPtrain = np.concatenate((inp_GPtrain, inpp.reshape(1,Ndim)))

            inp_LFtrain = np.concatenate((inp_LFtrain, inpp.reshape(1,Ndim)))
            y_HF_LFtrain = np.concatenate((y_HF_LFtrain, (y_nxt.reshape(1))))
            ML0 = ML_TF(obs_ind = Norm1(inp_LFtrain,P,Ndim), obs = (y_HF_LFtrain))
            amp0, len0 = ML0.GP_train(amp_init=1., len_init=1., num_iters = Iters)
            samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, pred_ind = Norm1(inp_GPtrain,P,Ndim), num_samples=num_s)
            y_LF_GP = np.array(InvNorm3(np.mean(np.array(samples0),axis=0),y_HF_LFtrain))
            y_HF_GP = np.concatenate((y_HF_GP, y_nxt.reshape(1)))
            y_GPtrain = y_HF_GP - y_LF_GP

            ML = ML_TF(obs_ind = Norm1(inp_GPtrain,P,Ndim), obs = (y_GPtrain))
            amp1, len1 = ML.GP_train(amp_init=1., len_init=1., num_iters = Iters)
            amp_sto = np.concatenate((amp_sto, np.array(amp1).reshape(1)))
            len_sto = np.concatenate((len_sto, np.array(len1).reshape(1)))
            subs_info[ii,kk] = 1.0

        if (y_nxt)>y1_lim[kk-1]:
            inp1[ii,:,kk] = inpp
            y1[ii,kk] = y_nxt
            LF_prev = LF
        else:
            inp1[ii,:,kk] = markov_seed
            y1[ii,kk] = markov_out
            Indicator[ii,kk] = 0.0
            ylf[ii,kk] = LF_prev

Pf = 1
Pi_sto = np.zeros(Nlim)
cov_sq = 0
for kk in np.arange(0,Nlim,1):
    Pi = len(np.rot90(np.where(y1[:,kk]>np.min([y1_lim[kk],value]))))/(Nsub)
    Pf = Pf * Pi
    Pi_sto[kk] = Pi
    cov_sq = cov_sq + ((1-Pi)/(Pi*Nsub))
cov_req = np.sqrt(cov_sq)
