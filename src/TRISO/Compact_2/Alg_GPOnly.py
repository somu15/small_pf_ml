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
P = np.array([2.7282643229905267e-05,5.230937638444862e-05,1.5642320384140967e-05,7.42526553558767e-06,2.0917204816879883e-05,1,1,1])
# P = np.array([1,1,1,1,1,1,1,1])

def Norm1(X1,X,dim):
    K = np.zeros((len(X1),dim))
    for ii in np.arange(0,dim,1):
        # K[:,ii] = np.reshape(((X1[:,ii])-np.mean((X[:,ii])))/(np.std((X[:,ii]))),len(X1))
        K[:,ii] = X1[:,ii]/X[ii]
    return K

def Norm3(X1):
    # return ((X1)-np.mean((X)))/(np.std((X)))
    return (X1/1.2e9) # 

def InvNorm3(X1):
    # return (X1*np.std((X))+np.mean((X)))
    return (X1*1.2e9) # 

Ninit_GP = 12
lhd = DR1.TrisoRandom(N=Ninit_GP) #  DR1.TrisoLHS(Nsamps=Ninit_GP) #  uniform(loc=-3.5,scale=7.0).ppf(lhd0) #
inp_GPtrain = lhd
y_HF_GP = np.array((LS1.Triso_1d(inp_GPtrain)))
# ind_req = np.where((y_HF_GP>-1.06e9) | (y_HF_GP<-1.07e9))
# y_HF_GP = y_HF_GP[ind_req]
# inp_GPtrain = inp_GPtrain[ind_req,:].reshape(len(y_HF_GP),7)
ML = ML_TF(obs_ind = Norm1(inp_GPtrain,P,Ndim), obs = Norm3(y_HF_GP))
amp0, len0 = ML.GP_train(amp_init=1.0, len_init=1.0, num_iters = 1000)
Iters = 300
samples0 = ML.GP_predict(amplitude_var = amp0, length_scale_var=len0, pred_ind = Norm1(inp_GPtrain,P,Ndim), num_samples=num_s)
y_LF_GP = np.array(InvNorm3(np.mean(np.array(samples0),axis=0)))

# DNN_model = ML.DNN_train(dim=Ndim, seed=100, neurons1=6, neurons2=4, learning_rate=0.002, epochs=2000)
# LF_DNN = ML.DNN_pred(inp_GPtrain,y_train,DNN_model,Ndim,inp_GPtest)[0].reshape(40)


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
    inp = DR1.TrisoRandom()
    inp = inp.reshape(Ndim)
    inpp = inp[None,:]
    samples0 = ML.GP_predict(amplitude_var = amp0, length_scale_var=len0, pred_ind = Norm1(inpp,P,Ndim), num_samples=num_s)
    LF = np.array(InvNorm3(np.mean(np.array(samples0),axis=0)))
    inp1[ii,:,0] = inp
    if ii <Ninit_GP:
        additive = np.percentile(y_HF_GP,90)
    else:
        additive = np.percentile(y1[0:ii,0],90)
    u_check = (np.abs(LF - additive))/np.std(InvNorm3(np.array(samples0)),axis=0)
    u_GP[ii,0] = u_check

    u_lim = u_lim_vec[0]
    print(ii)
    if u_check > u_lim:
        y1[ii,0] = LF
    else:
        y1[ii,0] = np.array(LS1.Triso_1d(inpp)).reshape(1)
        inp_GPtrain = np.concatenate((inp_GPtrain, inpp.reshape(1,Ndim)))
        y_LF_GP = np.concatenate((y_LF_GP, np.array(LF).reshape(1)))
        y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
        # ML = ML_TF(obs_ind = (np.array(inp_GPtrain))[:,:,0], obs = (np.array(y_HF_GP)[:,:,0]-np.array(y_LF_GP)[:,:,0])[:,0])
        ML = ML_TF(obs_ind = Norm1(inp_GPtrain,P,Ndim), obs = Norm3(y_HF_GP))
        amp0, len0 = ML.GP_train(amp_init=1., len_init=1., num_iters = Iters)
        subs_info[ii,0] = 1.0

ind_req = np.where(subs_info[:,0]==0.0)
samples0 = ML.GP_predict(amplitude_var = amp0, length_scale_var=len0, pred_ind = Norm1(inp1[ind_req,:,0].reshape(len(np.rot90(ind_req)),Ndim),P,Ndim), num_samples=500)
y1[ind_req,0] = np.array(InvNorm3(np.mean(np.array(samples0),axis=0)))

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

    inp_GPtrain = np.zeros((1,Ndim))
    y_LF_GP = np.empty(1, dtype = float)
    y_HF_GP = np.empty(1, dtype = float)
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

            rv1 = norm(loc=np.log(markov_seed[jj]),scale=std_prop[jj])
            prop = np.exp(rv1.rvs())
            r = np.log(DR1.TrisoPDF(rv_req=prop, index=jj)) - np.log(DR1.TrisoPDF(rv_req=(markov_seed[jj]),index=jj))
            if r>np.log(uni.rvs()):
                nxt[0,jj] = prop
            else:
                nxt[0,jj] = markov_seed[jj]
            inpp[0,jj] = nxt[0,jj]
        samples0 = ML.GP_predict(amplitude_var = amp0, length_scale_var=len0, pred_ind = Norm1(inpp,P,Ndim), num_samples=num_s)
        LF = np.array(InvNorm3(np.mean(np.array(samples0),axis=0)))
        if kk<(Nlim-1):
            if ii < Ninit_GP:
                additive =  y1_lim[kk-1]
            else:
                additive = np.percentile(y1[0:ii,kk],90) # y1_lim[kk-1]
        else:
            additive = value
        # additive = y1_lim[kk-1]
        u_check = (np.abs(LF - additive))/np.std(InvNorm3(np.array(samples0)),axis=0)
        u_GP[ii,kk] = u_check
        u_lim = u_lim_vec[kk]

        if u_check > u_lim and ii>=Ninit_GP:
            y_nxt = LF
        else:
            y_nxt = np.array(LS1.Triso_1d(inpp)).reshape(1)
            inp_GPtrain = np.concatenate((inp_GPtrain, inpp.reshape(1,Ndim)))
            y_LF_GP = np.concatenate((y_LF_GP, np.array(LF).reshape(1)))
            y_HF_GP = np.concatenate((y_HF_GP, y_nxt.reshape(1)))
            if ii == 0: # and gp_current==1:
                inp_GPtrain = np.delete(inp_GPtrain, 0, axis=0)
                y_LF_GP = np.delete(y_LF_GP, 0)
                y_HF_GP = np.delete(y_HF_GP, 0)
            ML = ML_TF(obs_ind = Norm1(inp_GPtrain,P,Ndim), obs = Norm3(y_HF_GP))
            amp0, len0 = ML.GP_train(amp_init=1., len_init=1., num_iters = Iters)
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
