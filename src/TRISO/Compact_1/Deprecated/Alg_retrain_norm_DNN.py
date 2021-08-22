
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
from statsmodels.distributions.empirical_distribution import ECDF

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
P = np.array([1,1,1,1,1,1,1,1])

## Training GP

def Norm1(X1,X,dim):
    K = np.zeros((len(X1),dim))
    for ii in np.arange(0,dim,1):
        K[:,ii] = np.reshape(((X1[:,ii])-np.mean((X[:,ii])))/(np.std((X[:,ii]))),len(X1))
    return K

rv_out = norm(loc=-962118758.436156, scale=207529923.79327065)
rv_norm = norm(loc=0,scale=1)

def Norm3(X1,X):
    return (X1-np.mean(X,axis=0))/(np.std(X,axis=0))

def InvNorm3(X1,X):
    return (X1*np.std(X,axis=0)+np.mean(X,axis=0))

## Train the LF model

Ninit_GP = 30
lhd0 = lhs(Ndim, samples=Ninit_GP, criterion='centermaximin')
lhd = uniform(loc=-3,scale=6).ppf(lhd0) # norm().ppf(lhd0) #
inp_LFtrain = lhd
y_HF_LFtrain = LS1.Triso_1d_norm(inp_LFtrain)
ML0 = ML_TF(obs_ind = inp_LFtrain, obs = y_HF_LFtrain) # , amp_init=1., len_init=1., var_init=1., num_iters = 1000)
DNN_model = ML0.DNN_train(dim=Ndim, seed=100, neurons1=10, neurons2=6, learning_rate=0.002, epochs=5000)

## Train the GP diff model

Ninit_GP = 12
lhd = DR1.StandardNormal_Indep(N=Ninit_GP)
inp_GPtrain = lhd
y_LF_GP = ML0.DNN_pred(inp_LFtrain,y_HF_LFtrain,DNN_model,Ndim,inp_GPtrain)[0].reshape(Ninit_GP)
y_HF_GP = np.array((LS1.Triso_1d_norm(inp_GPtrain)))
y_GPtrain = y_HF_GP - y_LF_GP
ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
amp1, len1 = ML.GP_train(amp_init=1., len_init=1., num_iters = 1000)
Iters = 400
amp_sto = np.array(amp1).reshape(1)
len_sto = np.array(len1).reshape(1)

# amp1, len1, mix1 = ML.GP_train_RQ(amp_init=1., len_init=1., mix_init=1., num_iters = 1000)

## Subset simultion with HF-LF and GP

uni = uniform()
Nsub = 250 # 10000
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
    inp = DR1.StandardNormal_Indep()
    inp = inp.reshape(Ndim)
    inpp = inp[None,:]
    LF = ML0.DNN_pred(inp_LFtrain,y_HF_LFtrain,DNN_model,Ndim,inpp)[0].reshape(1)
    inp1[ii,:,0] = inp
    if ii <Ninit_GP:
        additive = np.percentile(y_HF_GP,90)
    else:
        additive = np.percentile(y1[0:ii,0],90)
    samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim), num_samples=num_s)
    GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
    u_check = (np.abs(LF + GP_diff - additive))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
    u_GP[ii,0] = u_check

    u_lim = 2.0
    print(ii)
    if u_check > u_lim:
        y1[ii,0] = LF + GP_diff
    else:
        y1[ii,0] = (np.array((LS1.Triso_1d_norm(inpp))).reshape(1))
        inp_GPtrain = np.concatenate((inp_GPtrain, inpp.reshape(1,Ndim)))

        inp_LFtrain = np.concatenate((inp_LFtrain, inpp.reshape(1,Ndim)))
        y_HF_LFtrain = np.concatenate((y_HF_LFtrain, (y1[ii,0].reshape(1))))
        ML0 = ML_TF(obs_ind = inp_LFtrain, obs = y_HF_LFtrain)
        DNN_model = ML0.DNN_train(dim=Ndim, seed=100, neurons1=10, neurons2=6, learning_rate=0.002, epochs=5000)
        y_LF_GP = ML0.DNN_pred(inp_LFtrain,y_HF_LFtrain,DNN_model,Ndim,inp_GPtrain)[0].reshape(len(inp_GPtrain))
        y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
        y_GPtrain = y_HF_GP - y_LF_GP

        ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
        amp1, len1 = ML.GP_train(amp_init=1., len_init=len1, num_iters = Iters)
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
prop_std_req = np.array([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])

for kk in np.arange(1,Nlim,1):
    count = np.inf
    ind_max = 0
    ind_sto = -1
    seeds_outs = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
    y1_lim[kk-1] = np.min(seeds_outs)
    k = (y1[:,kk-1]).argsort()
    indices = k[int((1-Psub)*Nsub):(len(y1))]
    seeds = inp1[indices,:,kk-1]
    std_prop = ((seeds)).std(0)
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
        else:
            markov_seed = inp1[ii-1,:,kk]
            markov_out = y1[ii-1,kk]

        count = count + 1

        for jj in np.arange(0,Ndim,1):
            prop = markov_seed[jj] + (2*uni.rvs()-1)*1.0
            r = np.log(rv_norm.pdf(prop))-np.log(rv_norm.pdf(markov_seed[jj]))
            if r>np.log(uni.rvs()):
                nxt[0,jj] = prop
            else:
                nxt[0,jj] = markov_seed[jj]
            inpp[0,jj] = nxt[0,jj]

        LF = ML0.DNN_pred(inp_LFtrain,y_HF_LFtrain,DNN_model,Ndim,inpp)[0].reshape(1)
        samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim), num_samples=num_s)
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
            y_nxt = (np.array((LS1.Triso_1d_norm(inpp))).reshape(1))
            inp_GPtrain = np.concatenate((inp_GPtrain, inpp.reshape(1,Ndim)))

            inp_LFtrain = np.concatenate((inp_LFtrain, inpp.reshape(1,Ndim)))
            y_HF_LFtrain = np.concatenate((y_HF_LFtrain, (y_nxt.reshape(1))))
            ML0 = ML_TF(obs_ind = inp_LFtrain, obs = y_HF_LFtrain)
            DNN_model = ML0.DNN_train(dim=Ndim, seed=100, neurons1=10, neurons2=6, learning_rate=0.002, epochs=5000)
            y_LF_GP = ML0.DNN_pred(inp_LFtrain,y_HF_LFtrain,DNN_model,Ndim,inp_GPtrain)[0].reshape(len(inp_GPtrain))
            y_HF_GP = np.concatenate((y_HF_GP, y_nxt.reshape(1)))
            y_GPtrain = y_HF_GP - y_LF_GP

            ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
            amp1, len1 = ML.GP_train(amp_init=1., len_init=len1, num_iters = Iters)
            amp_sto = np.concatenate((amp_sto, np.array(amp1).reshape(1)))
            len_sto = np.concatenate((len_sto, np.array(len1).reshape(1)))
            subs_info[ii,kk] = 1.0

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
