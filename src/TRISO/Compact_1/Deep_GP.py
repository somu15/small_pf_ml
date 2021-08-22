#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 23:17:57 2021

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

# import gpflow
# from gpflow.utilities import print_summary, set_trainable
# from gpflow.models import VGP, GPR, SGPR, SVGP
# from gpflow.optimizers import NaturalGradient
# from gpflow.optimizers.natgrad import XiSqrtMeanVar
# # from gpflow import config
# gpflow.config.set_default_jitter(1e-4)

Ndim = 8
value = 0.0

LS1 = LSF()
DR1 = DR()
num_s = 500
P = np.array([213.35e-6,98.9e-6,40.4e-6,35.2e-6,43.4e-6,1,1,1])

## Training GP

def Norm1(X1,X,dim):
    K = np.zeros((len(X1),dim))
    for ii in np.arange(0,dim,1):
        K[:,ii] = np.reshape(((X1[:,ii])-np.mean((X[:,ii])))/(np.std((X[:,ii]))),len(X1))
    return K

rv_out = norm(loc=191023345.13968003, scale=282810513.3103714)
rv_norm = norm(loc=0,scale=1)
jitter_std = 0.1
jitter = norm(loc=0,scale=496994.68*jitter_std)
std_jitter = 0.001

def Norm3(X1,X):
    return (X1-np.mean(X,axis=0))/(np.std(X,axis=0)) # /4031437.0841159956 # /191023345.13968003 # 

def InvNorm3(X1,X):
    return (X1*np.std(X,axis=0)+np.mean(X,axis=0)) # *4031437.0841159956 # *191023345.13968003 # 

## Train the LF model

Ninit_GP = 12
lhd0 = lhs(Ndim, samples=Ninit_GP, criterion='centermaximin') # DR1.StandardNormal_Indep(N=Ninit_GP) # 
lhd = uniform(loc=-3,scale=6).ppf(lhd0) # norm().ppf(lhd0) #
inp_LFtrain = lhd
y_HF_LFtrain = LS1.Triso_1d_norm(inp_LFtrain)
# y_HF_LFtrain = tmp_1
ML0 = ML_TF(obs_ind = inp_LFtrain, obs = y_HF_LFtrain+norm(loc=0,scale=np.std(y_HF_LFtrain)*std_jitter).rvs(len(y_HF_LFtrain)))
DNN_model = ML0.DNN_train(dim=Ndim, neurons1=10, neurons2=6, learning_rate=0.002, epochs=5000)

## Train the GP diff model

Ninit_GP = 12
lhd = DR1.StandardNormal_Indep(N=Ninit_GP)
inp_GPtrain = uniform(loc=-2,scale=4).ppf(lhd0) # lhd # inp_LFtrain # 
y_LF_GP = ML0.DNN_pred(inp_LFtrain,y_HF_LFtrain,DNN_model,Ndim,inp_GPtrain)[0].reshape(Ninit_GP) # InvNorm3(pce2.predict(Norm1(inp_GPtrain,inp_LFtrain,Ndim)).reshape(Ninit_GP),y_HF_LFtrain) #   
y_HF_GP = np.array((LS1.Triso_1d_norm(inp_GPtrain))) # y_HF_LFtrain # 
y_GPtrain = y_HF_GP - y_LF_GP
ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
amp1, len1 = ML.GP_train_kernel(amp_init=np.float64(1.), len_init=np.float64([1.]*inp_GPtrain.shape[1]), num_iters = 1000)

iters = 400
## Subset simultion with HF-LF and GP

uni = uniform()
Nsub = 1000 # 10000
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

u_check = 0.0
inp = DR1.StandardNormal_Indep(N=Nsub)
inpp = inp[None,:]
inp1[:,:,0] = inp
LF = ML0.DNN_pred(inp_LFtrain,y_HF_LFtrain,DNN_model,Ndim,inp)[0].reshape(Nsub)
samples1 = ML.GP_predict_kernel(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inp,inp_GPtrain,Ndim), num_samples=num_s)
GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
GP_std = np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
y1[:,0] = LF+GP_diff
additive = np.percentile(y1[:,0],90)
u_all = (np.abs(LF + GP_diff - additive))/GP_std
while u_check < 2:
    u_check = np.min(u_all)
    ind_req = np.where(u_all==np.min(u_all))
    subs_info[ind_req,0] = 1
    inp_req = inp[ind_req,:].reshape(1,Ndim)
    y1[ind_req,0] = (np.array((LS1.Triso_1d_norm(inp_req))).reshape(1))
    inp_GPtrain = np.concatenate((inp_GPtrain, inp_req))
    y_LF_GP = ML0.DNN_pred(inp_LFtrain,y_HF_LFtrain,DNN_model,Ndim,inp_GPtrain)[0].reshape(len(inp_GPtrain))
    y_HF_GP = np.concatenate((y_HF_GP, y1[ind_req,0].reshape(1)))
    y_GPtrain = y_HF_GP - y_LF_GP
    tmp1,i = np.unique(y_GPtrain,return_index=True)
    y = y_GPtrain[i]
    x = inp_GPtrain[i,:]
    ML = ML_TF(obs_ind = Norm1(x,x,Ndim), obs = Norm3(y,y))
    amp1, len1 = ML.GP_train_kernel(amp_init=amp1, len_init=len1, num_iters = iters)
    
    ind_req1 = np.where(subs_info[:,0]==0)
    LF = ML0.DNN_pred(inp_LFtrain,y_HF_LFtrain,DNN_model,Ndim,inp[ind_req1,:].reshape(len(np.rot90(ind_req1)),Ndim))[0].reshape(len(np.rot90(ind_req1)))
    samples1 = ML.GP_predict_kernel(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inp[ind_req1,:].reshape(len(np.rot90(ind_req1)),Ndim),inp_GPtrain,Ndim), num_samples=num_s)
    GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
    GP_std = np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
    y1[ind_req1,0] = LF+GP_diff
    additive = np.percentile(y1[:,0],90)
    u_all[ind_req1] = (np.abs(LF + GP_diff - additive))/GP_std
    ind_req2 = np.where(subs_info[:,0]==1)
    u_all[ind_req2] = 100.0
    



flag_DNN = 0

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
    
    samples1 = ML.GP_predict_kernel(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim), num_samples=num_s)
    GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
    GP_std = np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
    u_check = (np.abs(LF + GP_diff - additive))/GP_std
    u_GP[ii,0] = u_check

    u_lim = 2.0
    print(ii)
    if u_check > u_lim:
        y1[ii,0] = LF + GP_diff
    else:
        
        y1[ii,0] = (np.array((LS1.Triso_1d_norm(inpp))).reshape(1))
        
        # inp_LFtrain = np.concatenate((inp_LFtrain, inp.reshape(1,Ndim)))
        # y_HF_LFtrain = np.concatenate((y_HF_LFtrain, (y1[ii,0].reshape(1))))
        # ML0 = ML_TF(obs_ind = inp_LFtrain, obs = y_HF_LFtrain+norm(loc=0,scale=np.std(y_HF_LFtrain)*std_jitter).rvs(len(y_HF_LFtrain)))
        # DNN_model = ML0.DNN_train(dim=Ndim, neurons1=10, neurons2=6, learning_rate=0.002, epochs=4000)
        # inp_GPtrain = np.concatenate((inp_GPtrain, inpp.reshape(1,Ndim)))
        # y_LF_GP = ML0.DNN_pred(inp_LFtrain,y_HF_LFtrain,DNN_model,Ndim,inp_GPtrain)[0].reshape(len(inp_GPtrain))
        # y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
        # y_GPtrain = y_HF_GP - y_LF_GP #  + jitter.rvs(len(y_HF_GP))
        # tmp1,i = np.unique(y_GPtrain,return_index=True)
        # y = y_GPtrain[i]
        # x = inp_GPtrain[i,:]
        # ML = ML_TF(obs_ind = Norm1(x,x,Ndim), obs = Norm3(y,y))
        # amp1, len1 = ML.GP_train_kernel(amp_init=amp1, len_init=len1, num_iters = iters)
        
        if flag_DNN==100000: 
            inp_LFtrain = np.concatenate((inp_LFtrain, inp.reshape(1,Ndim)))
            y_HF_LFtrain = np.concatenate((y_HF_LFtrain, (y1[ii,0].reshape(1))))
            ML0 = ML_TF(obs_ind = inp_LFtrain, obs = y_HF_LFtrain) # +norm(loc=0,scale=np.std(y_HF_LFtrain)*std_jitter).rvs(len(y_HF_LFtrain))
            DNN_model = ML0.DNN_train(dim=Ndim, neurons1=10, neurons2=6, learning_rate=0.002, epochs=4000)
            y_LF_GP = ML0.DNN_pred(inp_LFtrain,y_HF_LFtrain,DNN_model,Ndim,inp_GPtrain)[0].reshape(len(inp_GPtrain))
            y_GPtrain = y_HF_GP - y_LF_GP #  + jitter.rvs(len(y_HF_GP))
            tmp1,i = np.unique(y_GPtrain,return_index=True)
            y = y_GPtrain[i]
            x = inp_GPtrain[i,:]
            ML = ML_TF(obs_ind = Norm1(x,x,Ndim), obs = Norm3(y,y))
            amp1, len1 = ML.GP_train_kernel(amp_init=amp1, len_init=len1, num_iters = iters)
            flag_DNN = 0
            # flag_GP = 0
        else: 
            inp_GPtrain = np.concatenate((inp_GPtrain, inpp.reshape(1,Ndim)))
            y_LF_GP = ML0.DNN_pred(inp_LFtrain,y_HF_LFtrain,DNN_model,Ndim,inp_GPtrain)[0].reshape(len(inp_GPtrain))
            y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
            y_GPtrain = y_HF_GP - y_LF_GP #  + jitter.rvs(len(y_HF_GP))
            tmp1,i = np.unique(y_GPtrain,return_index=True)
            y = y_GPtrain[i]
            x = inp_GPtrain[i,:]
            ML = ML_TF(obs_ind = Norm1(x,x,Ndim), obs = Norm3(y,y))
            amp1, len1 = ML.GP_train_kernel(amp_init=amp1, len_init=len1, num_iters = iters)
            flag_DNN = flag_DNN + 1
            # flag_GP = 1
            
        
        subs_info[ii,0] = 1.0

