#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 10:40:21 2021

@author: dhulls
"""

from os import sys
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
value = 0.0 # 250.0

LS1 = LSF()
DR1 = DR()
num_s = 500
P = np.array([1,1,1,1,1,1,1,1])

rv_norm = norm(loc=0,scale=1)

def Norm1(X1,X,dim):
    K = np.zeros((len(X1),dim))
    for ii in np.arange(0,dim,1):
        K[:,ii] = np.reshape(((X1[:,ii])-np.mean((X[:,ii])))/(np.std((X[:,ii]))),len(X1))
    return K

def Norm3(X1,X):
    return (X1-np.mean(X,axis=0))/(np.std(X,axis=0))

def InvNorm3(X1,X):
    return (X1*np.std(X,axis=0)+np.mean(X,axis=0))

Ninit_GP = 12
lhd = DR1.StandardNormal_Indep(N=Ninit_GP)
inp_GPtrain = lhd
y_HF_GP = np.array((LS1.Triso_1d_norm(inp_GPtrain)))
ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_HF_GP,y_HF_GP))
amp0, len0 = ML.GP_train_kernel(num_iters = 1000)
Iters = 300
samples0 = ML.GP_predict_kernel(amplitude_var = amp0, length_scale_var=len0, pred_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), num_samples=num_s)
y_LF_GP = np.array(InvNorm3(np.mean(np.array(samples0),axis=0),y_HF_GP))

uni = uniform()
Nsub = 1000
Psub = 0.1
Nlim = 4
y1 = np.zeros((Nsub,Nlim))
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

for ii in np.arange(0,Nsub,1):
    inp = DR1.StandardNormal_Indep()
    inp = inp.reshape(Ndim)
    inpp = inp[None,:]
    samples0 = ML.GP_predict_kernel(amplitude_var = amp0, length_scale_var=len0, pred_ind = Norm1(inpp,inp_GPtrain,Ndim), num_samples=num_s)
    LF = np.array(InvNorm3(np.mean(np.array(samples0),axis=0),y_HF_GP))
    inp1[ii,:,0] = inp
    if ii <Ninit_GP:
        additive = np.percentile(y_HF_GP,90)
    else:
        additive = np.percentile(y1[0:ii,0],90)
    u_check = (np.abs(LF - additive))/np.std(InvNorm3(np.array(samples0),y_HF_GP),axis=0)
    u_GP[ii,0] = u_check

    u_lim = u_lim_vec[0]
    print(ii)
    if u_check > u_lim:
        y1[ii,0] = LF
    else:
        y1[ii,0] = np.array(LS1.Triso_1d_norm(inpp)).reshape(1)
        inp_GPtrain = np.concatenate((inp_GPtrain, inpp.reshape(1,Ndim)))
        y_LF_GP = np.concatenate((y_LF_GP, np.array(LF).reshape(1)))
        y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
        ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_HF_GP,y_HF_GP))
        amp0, len0 = ML.GP_train_kernel(num_iters = Iters)
        subs_info[ii,0] = 1.0

ind_req = np.where(subs_info[:,0]==0.0)
samples0 = ML.GP_predict_kernel(amplitude_var = amp0, length_scale_var=len0, pred_ind = Norm1(inp1[ind_req,:,0].reshape(len(np.rot90(ind_req)),Ndim),inp_GPtrain,Ndim), num_samples=500)
y1[ind_req,0] = np.array(InvNorm3(np.mean(np.array(samples0),axis=0),y_HF_GP))

LF_plus_GP = np.delete(LF_plus_GP, 0)
GP_pred = np.delete(GP_pred, 0)

inpp = np.zeros((1,Ndim))
count_max = int(Nsub/(Psub*Nsub))-1
seeds_outs = np.zeros(int(Psub*Nsub))
seeds = np.zeros((int(Psub*Nsub),Ndim))
markov_seed = np.zeros(Ndim)
markov_out = 0.0
u_req = np.zeros(Nsub)
u_check1 = 10.0
std_prop = np.zeros(Ndim)
Indicator = np.ones((Nsub,Nlim))

prop_std_req = np.array([0.0216,0.75,11373.07,25.98,11.453,25.98,121.243,474.148])

for kk in np.arange(1,Nlim,1):

    # inp_GPtrain = np.zeros((1,Ndim))
    # y_LF_GP = np.empty(1, dtype = float)
    # y_HF_GP = np.empty(1, dtype = float)
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

            prop = markov_seed[jj] + (2*uni.rvs()-1)*1.0 # std_prop[jj]
            r = np.log(rv_norm.pdf(prop))-np.log(rv_norm.pdf(markov_seed[jj]))
            if r>np.log(uni.rvs()):
                nxt[0,jj] = prop
            else:
                nxt[0,jj] = markov_seed[jj]
            inpp[0,jj] = nxt[0,jj]
        samples0 = ML.GP_predict_kernel(amplitude_var = amp0, length_scale_var=len0, pred_ind = Norm1(inpp,inp_GPtrain,Ndim), num_samples=num_s)
        LF = np.array(InvNorm3(np.mean(np.array(samples0),axis=0),y_HF_GP))
        if kk<(Nlim-1):
            if ii < Ninit_GP:
                additive =  y1_lim[kk-1]
            else:
                additive = np.percentile(y1[0:ii,kk],90)
        else:
            additive = value
        u_check = (np.abs(LF - additive))/np.std(InvNorm3(np.array(samples0),y_HF_GP),axis=0)
        u_GP[ii,kk] = u_check
        u_lim = u_lim_vec[kk]

        if u_check > u_lim and ii>=Ninit_GP:
            y_nxt = LF
        else:
            y_nxt = np.array(LS1.Triso_1d_norm(inpp)).reshape(1)
            inp_GPtrain = np.concatenate((inp_GPtrain, inpp.reshape(1,Ndim)))
            y_LF_GP = np.concatenate((y_LF_GP, np.array(LF).reshape(1)))
            y_HF_GP = np.concatenate((y_HF_GP, y_nxt.reshape(1)))
            ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_HF_GP,y_HF_GP))
            amp0, len0 = ML.GP_train_kernel(num_iters = Iters)
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
