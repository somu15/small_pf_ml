#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 12:04:41 2021

@author: dhulls
"""

from os import sys
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import random
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import rayleigh
from scipy.stats import uniform
from scipy.stats import cauchy
import matplotlib.pyplot as plt
from UQpy.SampleMethods import MH
from UQpy.Distributions import Distribution
import time
from UQpy.Distributions import Normal
from UQpy.SampleMethods import MMH

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

Ndim = 2
value = 0.0

LS1 = LSF()
DR1 = DR()
num_s = 500

def Convert(lst): 
    return [ -i for i in lst ] 

## Monte Carlo simulations

# Nsims = int(5e6)
# y = np.zeros(Nsims)
# ys = np.zeros(Nsims)
# LS1 = LSF()
# DR1 = DR()
# Ndim = 8
# value = 250

# for ii in np.arange(0,Nsims,1):
#     inp = (DR1.BoreholeRandom())
#     inpp = inp[None,:]
#     y[ii] = np.array(LS1.Scalar_Borehole_HF_nD(inpp))

# req = len(np.rot90(np.where(y>value)))/Nsims

# req = 0.0003414 (Borehole; 5e6 simulations)

# Basic subset simulation

# LS1 = LSF()
# DR1 = DR()
# num_s = 500

# uni = uniform()
# Nsub = 2000
# Psub = 0.1
# Nlim = 3
# y1 = np.zeros((Nsub,Nlim))
# y1_lim = np.zeros(Nlim)
# y1_lim[Nlim-1] = value
# inp1 = np.zeros((Nsub,Ndim,Nlim))
# rv = norm(loc=0,scale=1)
# y_seed = np.zeros(int(Psub*Nsub))

# for ii in np.arange(0,Nsub,1):
#     inp = DR1.StandardNormal_Indep(N=Ndim)
#     print(ii)
#     inpp = inp[None,:]
#     y1[ii,0] = np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp))).reshape(1)
#     inp1[ii,:,0] = inp

# inpp = np.zeros(Ndim)
# count_max = Nsub/(Psub*Nsub)
# count = 100000
# ind_max = 1
# r_sto = np.zeros((Nsub-int(Psub*Nsub),Nlim-1,Ndim))
# # ind_sto = 3

# for kk in np.arange(1,Nlim,1):
#     ind_max = 0
#     ind_sto = -1
#     count = np.inf
#     y1[0:(int(Psub*Nsub)),kk] = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
#     y1_lim[kk-1] = np.min(y1[0:(int(Psub*Nsub)),kk])
#     indices = (-y1[:,kk-1]).argsort()[:(int(Psub*Nsub))]
#     inp1[0:(int(Psub*Nsub)),:,kk] = inp1[indices,:,kk-1]
#     for ii in np.arange((int(Psub*Nsub)),(Nsub),1):
#         print(kk)
#         print(ii)
#         nxt = np.zeros((1,Ndim))
#         if count > count_max:
#             # ind_max = random.randint(0,int(Psub*Nsub)) # ind_sto
#             ind_sto = ind_sto + 1
#             ind_max = ind_sto
#             count = 0
#         else:
#             ind_max = ii-1
            
#         count = count + 1

#         for jj in np.arange(0,Ndim,1):
#             # if jj == 0:
#             #     rv1 = norm(loc=inp1[ind_max,jj,kk],scale=1.0)
#             # else:
#             #     rv1 = norm(loc=inp1[ind_max,jj,kk],scale=1.0)
#             rv1 = norm(loc=(inp1[ind_max,jj,kk]),scale=1.0)
#             prop = (rv1.rvs())
#             r = np.log(norm.pdf(prop)) - np.log(norm.pdf(inp1[ind_max,jj,kk])) # rv.pdf((prop))/rv.pdf((inp1[ii-(int(Psub*Nsub)),jj,kk]))
#             r_sto[ii-(int(Psub*Nsub)),kk-1,jj] = r
#             if r>np.log(uni.rvs()):
#                 nxt[0,jj] = prop
#             else: 
#                 nxt[0,jj] = inp1[ind_max,jj,kk]
#             inpp[jj] = nxt[0,jj]
#         y_nxt = np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp[None,:]))).reshape(1)
#         if y_nxt>y1_lim[kk-1]:
#             inp1[ii,:,kk] = inpp # np.array([nxt[0,0], nxt[0,1], nxt[0,2]])
#             y1[ii,kk] = y_nxt
#         else:
#             inp1[ii,:,kk] = inp1[ind_max,:,kk]
#             y1[ii,kk] = y1[ind_max,kk]

# Pf = 1
# Pi_sto = np.zeros(Nlim)
# cov_sq = 0
# for kk in np.arange(0,Nlim,1):
#     Pi = len(np.rot90(np.where(y1[:,kk]>np.min([y1_lim[kk],value]))))/(Nsub)
#     Pf = Pf * Pi
#     Pi_sto[kk] = Pi
#     cov_sq = cov_sq + ((1-Pi)/(Pi*Nsub))
# cov_req = np.sqrt(cov_sq)

## Pf = 0.0003651 [value = 250.0; cov_req = 0.0437; 5 subsets with 15000 sims per subset]
## Pf = 4.9881177e-05 [value = 270.0; cov_req = 0.0786; 5 subsets with 6000 sims per subset]

## Training GP

def Norm1(X1,X,dim):
    K = np.zeros((len(X1),dim))
    for ii in np.arange(0,dim,1):
        K[:,ii] = np.reshape((X1[:,ii]-np.mean(X[:,ii]))/(np.std(X[:,ii])),len(X1))
    return K

def Norm2(X1,X):
    return (np.log(X1)-np.mean(np.log(X)))/(np.std(np.log(X)))

# def InvNorm1(X1,X):
#     return X1 # (X1*np.std(X,axis=0)+np.mean(X,axis=0))

def InvNorm2(X1,X):
    return np.exp(X1*np.std(np.log(X))+np.mean(np.log(X)))


def Norm3(X1,X):
    return ((X1)-np.mean((X)))/(np.std((X)))

def InvNorm3(X1,X):
    return (X1*np.std((X))+np.mean((X)))

def Norm1(X1,X,dim):
    K = np.zeros((len(X1),dim))
    for ii in np.arange(0,dim,1):
        K[:,ii] = np.reshape(((X1[:,ii])-np.mean((X[:,ii])))/(np.std((X[:,ii]))),len(X1))
    return K

def Norm2(X1,X):
    return ((X1)-np.mean((X)))/(np.std((X)))

# def InvNorm1(X1,X):
#     return X1 # (X1*np.std(X,axis=0)+np.mean(X,axis=0))

def InvNorm2(X1,X):
    return np.exp(X1*np.std((X))+np.mean((X)))


def Norm3(X1,X):
    return ((X1)-np.mean((X)))/(np.std((X)))

def InvNorm3(X1,X):
    return (X1*np.std((X))+np.mean((X)))

Ninit_GP = 12
lhd0 = lhs(2, samples=Ninit_GP, criterion='centermaximin')
lhd = uniform(loc=-4,scale=8).ppf(lhd0)
inp_GPtrain = lhd
y_GPtrain = np.array(Convert(LS1.Scalar_LS1_HF_2D(inp_GPtrain)))
ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
amp1, len1, var1 = ML.GP_train(amp_init=1., len_init=1., var_init=1., num_iters = 1000)
Iters = 300

# ## Subset simultion with HF-LF and GP

uni = uniform()
Nsub = 1000
Psub = 0.1
Nlim = 3
y1 = np.zeros((Nsub,Nlim))
y1_lim = np.zeros(Nlim)
y1_lim[Nlim-1] = value
inp1 = np.zeros((Nsub,Ndim,Nlim))
rv = norm(loc=0,scale=1)
u_lim_vec = np.array([2,2,2,2,2,2,2,2,2])

u_GP = np.empty(1, dtype = float)
var_GP = np.empty(1, dtype = float)
std_GPdiff = np.empty(1, dtype = float)
var_GP[0] = var1.numpy().reshape(1)
subs_info = np.empty(1, dtype = float)
subs_info[0] = np.array(0).reshape(1)
LF_plus_GP = np.empty(1, dtype = float)
GP_pred = np.empty(1, dtype = float)
additive = value

for ii in np.arange(0,Nsub,1):
    inp = DR1.StandardNormal_Indep(N=Ndim)
    inpp = inp[None,:]
    samples0 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = inpp, num_samples=num_s)
    GP = np.array(np.mean(np.array(samples0),axis=0))
    if ii > 9:
        additive = np.percentile(y1[1:ii,0],90)
    u_check = (np.abs(GP - additive))/np.std(np.array(samples0),axis=0)
    u_GP = np.concatenate((u_GP, np.array(u_check).reshape(1)))
    std_GPdiff = np.concatenate((std_GPdiff, np.array(np.std(np.array(samples0),axis=0)).reshape(1)))
    u_lim = u_lim_vec[0]
    print(ii)
    if u_check > u_lim:
        y1[ii,0] = GP
    else:
        y1[ii,0] = np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp))).reshape(1)
        inp_GPtrain = np.concatenate((inp_GPtrain, inp.reshape(1,Ndim)))
        y_GPtrain = np.concatenate((y_GPtrain, y1[ii,0].reshape(1)))
        GP_pred = np.concatenate((GP_pred, (np.array(GP).reshape(1))))
        ML = ML_TF(obs_ind = inp_GPtrain, obs = y_GPtrain)
        amp1, len1, var1 = ML.GP_train(amp_init=amp1, len_init=len1, var_init=var1, num_iters = Iters)
        var_GP = np.concatenate((var_GP, var1.numpy().reshape(1)))
        subs_info = np.concatenate((subs_info, np.array(0).reshape(1)))
        
u_GP = np.delete(u_GP, 0)
var_GP = np.delete(var_GP, 0)
std_GPdiff = np.delete(std_GPdiff, 0)
subs_info = np.delete(subs_info, 0)

count_max = int(Nsub/(Psub*Nsub))

for kk in np.arange(1,Nlim,1):
    count = np.inf
    ind_max = 0
    ind_sto = -1
    y1[0:(int(Psub*Nsub)),kk] = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
    y1_lim[kk-1] = np.min(y1[0:(int(Psub*Nsub)),kk])
    indices = (-y1[:,kk-1]).argsort()[:(int(Psub*Nsub))]
    inp1[0:(int(Psub*Nsub)),:,kk] = inp1[indices,:,kk-1]
    for ii in np.arange((int(Psub*Nsub)),(Nsub),1):
        print(kk)
        print(ii)
        nxt = np.zeros((1,Ndim))
        
        if count > count_max:
            # ind_max = random.randint(0,int(Psub*Nsub))
            ind_sto = ind_sto + 1
            ind_max = ind_sto
            count = 0
        else:
            ind_max = ii-1
            
        count = count + 1
        
        for jj in np.arange(0,Ndim,1):
            # if jj == 0:
            #     rv1 = norm(loc=inp1[ind_max,jj,kk],scale=1.0)
            # else:
            #     rv1 = norm(loc=inp1[ind_max,jj,kk],scale=1.0)
            # rv1 = norm(loc=(inp1[ind_max,jj,kk]),scale=0.05)
            prop = (rv1.rvs())
            rv00 = norm(loc=0,scale=1)
            r = np.log(rv00.pdf(prop)) - np.log(rv00.pdf(inp1[ind_max,jj,kk]))
            if r>np.log(uni.rvs()):
                nxt[0,jj] = prop
            else: 
                nxt[0,jj] = inp1[ind_max,jj,kk]
            inpp[0,jj] = nxt[0,jj]
        samples0 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = inpp[None,:], num_samples=num_s)
        GP = np.array(np.mean(np.array(samples0),axis=0))
        if ii > (int(Psub*Nsub)+9): # and kk < (Nlim-1):
            additive = np.percentile(y1[1:ii,kk],90)
        else:
            additive = value
        u_check = (np.abs(GP - additive))/np.std(np.array(samples0),axis=0)
        u_GP = np.concatenate((u_GP, np.array(u_check).reshape(1)))
        std_GPdiff = np.concatenate((std_GPdiff, np.array(np.std(np.array(samples0),axis=0)).reshape(1)))
        u_lim = u_lim_vec[kk]
        if u_check > u_lim: # and ii > (int(Psub*Nsub)+num_retrain):
            y_nxt = GP
        else:
            y_nxt = np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp))).reshape(1)
            inp_GPtrain = np.concatenate((inp_GPtrain, inp.reshape(1,Ndim)))
            y_GPtrain = np.concatenate((y_GPtrain, y_nxt.reshape(1)))
            GP_pred = np.concatenate((GP_pred, (np.array(GP).reshape(1))))
            ML = ML_TF(obs_ind = inp_GPtrain, obs = y_GPtrain)
            amp1, len1, var1 = ML.GP_train(amp_init=amp1, len_init=len1, var_init=var1, num_iters = Iters)
            var_GP = np.concatenate((var_GP, var1.numpy().reshape(1)))
            subs_info = np.concatenate((subs_info, np.array(0).reshape(1)))
            
        if (y_nxt)>y1_lim[kk-1]:
            inp1[ii,:,kk] = inpp
            y1[ii,kk] = y_nxt
        else:
            inp1[ii,:,kk] = inp1[ind_max,:,kk]
            y1[ii,kk] = y1[ind_max,kk]

y1_lim[Nlim-1] = value
Pf = 1
Pi_sto = np.zeros(Nlim)
cov_sq = 0
for kk in np.arange(0,Nlim,1):
    Pi = len(np.rot90(np.where(y1[:,kk]>np.min([y1_lim[kk],value]))))/(Nsub)
    Pf = Pf * Pi
    Pi_sto[kk] = Pi
    cov_sq = cov_sq + ((1-Pi)/(Pi*Nsub))
cov_req = np.sqrt(cov_sq)

# uni = uniform()
# Nsub = 1000
# Psub = 0.1
# Nlim = 3
# y1 = np.zeros((Nsub,Nlim))
# y1_lim = np.zeros(Nlim)
# y1_lim[Nlim-1] = value
# inp1 = np.zeros((Nsub,Ndim,Nlim))
# rv = norm(loc=0,scale=1)
# u_lim_vec = np.array([2,2,2,2,2,2,2,2,2])

# u_GP = np.empty(1, dtype = float)
# var_GP = np.empty(1, dtype = float)
# std_GPdiff = np.empty(1, dtype = float)
# var_GP[0] = var1.numpy().reshape(1)
# subs_info = np.empty(1, dtype = float)
# subs_info[0] = np.array(0).reshape(1)
# LF_plus_GP = np.empty(1, dtype = float)
# GP_pred = np.empty(1, dtype = float)
# additive = value

# for ii in np.arange(0,Nsub,1):
#     inp = DR1.StandardNormal_Indep(N=Ndim)
#     inpp = inp[None,:]
#     samples0 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim), num_samples=num_s)
#     GP = np.array(InvNorm3(np.mean(np.array(samples0),axis=0),y_GPtrain))
#     if ii > 9:
#         additive = np.percentile(y1[1:ii,0],90)
#     u_check = np.abs(np.abs(GP) - np.abs(additive))/np.std(InvNorm3(np.array(samples0),y_GPtrain),axis=0)
#     u_GP = np.concatenate((u_GP, np.array(u_check).reshape(1)))
#     std_GPdiff = np.concatenate((std_GPdiff, np.array(np.std(InvNorm3(np.array(samples0),y_GPtrain),axis=0)).reshape(1)))
#     u_lim = u_lim_vec[0]
#     print(ii)
#     if u_check > u_lim:
#         y1[ii,0] = GP
#     else:
#         y1[ii,0] = np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp))).reshape(1)
#         inp_GPtrain = np.concatenate((inp_GPtrain, inp.reshape(1,Ndim)))
#         y_GPtrain = np.concatenate((y_GPtrain, y1[ii,0].reshape(1)))
#         GP_pred = np.concatenate((GP_pred, (np.array(GP).reshape(1))))
#         ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
#         amp1, len1, var1 = ML.GP_train(amp_init=amp1, len_init=len1, var_init=var1, num_iters = Iters)
#         var_GP = np.concatenate((var_GP, var1.numpy().reshape(1)))
#         subs_info = np.concatenate((subs_info, np.array(0).reshape(1)))
        
# u_GP = np.delete(u_GP, 0)
# var_GP = np.delete(var_GP, 0)
# std_GPdiff = np.delete(std_GPdiff, 0)
# subs_info = np.delete(subs_info, 0)

# count_max = int(Nsub/(Psub*Nsub))

# for kk in np.arange(1,Nlim,1):
#     count = np.inf
#     ind_max = 0
#     ind_sto = -1
#     y1[0:(int(Psub*Nsub)),kk] = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
#     y1_lim[kk-1] = np.min(y1[0:(int(Psub*Nsub)),kk])
#     indices = (-y1[:,kk-1]).argsort()[:(int(Psub*Nsub))]
#     inp1[0:(int(Psub*Nsub)),:,kk] = inp1[indices,:,kk-1]
#     for ii in np.arange((int(Psub*Nsub)),(Nsub),1):
#         print(kk)
#         print(ii)
#         nxt = np.zeros((1,Ndim))
        
#         if count > count_max:
#             # ind_max = random.randint(0,int(Psub*Nsub))
#             ind_sto = ind_sto + 1
#             ind_max = ind_sto
#             count = 0
#         else:
#             ind_max = ii-1
            
#         count = count + 1
        
#         for jj in np.arange(0,Ndim,1):
#             # if jj == 0:
#             #     rv1 = norm(loc=inp1[ind_max,jj,kk],scale=1.0)
#             # else:
#             #     rv1 = norm(loc=inp1[ind_max,jj,kk],scale=1.0)
#             # rv1 = norm(loc=(inp1[ind_max,jj,kk]),scale=1.0)
#             rv1 = uniform(loc=(inp1[ind_max,jj,kk]-0.75),scale=(2*0.75))
#             prop = (rv1.rvs())
#             r = np.log(norm.pdf(prop)) - np.log(norm.pdf(inp1[ind_max,jj,kk])) # np.log(rv.pdf((prop)))-np.log(rv.pdf((inp1[ind_max,jj,kk])))
#             if r>np.log(uni.rvs()):
#                 nxt[0,jj] = prop
#             else: 
#                 nxt[0,jj] = inp1[ind_max,jj,kk]
#             inpp[0,jj] = nxt[0,jj]
#         # samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, observation_noise_variance_var=var0, pred_ind = Norm1(inpp,inp_LFtrain,Ndim), num_samples=num_s)
#         # LF = np.array(InvNorm2(np.mean(np.array(samples0),axis=0),y_HF_LFtrain))
#         samples0 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim), num_samples=num_s)
#         GP = np.array(InvNorm3(np.mean(np.array(samples0),axis=0),y_GPtrain))
#         if ii > (int(Psub*Nsub)+9) and kk < (Nlim-1):
#             additive = np.percentile(y1[1:ii,kk],90)
#         else:
#             additive = value
#         u_check = np.abs(np.abs(GP) - np.abs(additive))/np.std(InvNorm3(np.array(samples0),y_GPtrain),axis=0)
#         # u_check = (np.abs(LF + GP_diff - value))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
#         u_GP = np.concatenate((u_GP, np.array(u_check).reshape(1)))
#         std_GPdiff = np.concatenate((std_GPdiff, np.array(np.std(InvNorm3(np.array(samples0),y_GPtrain),axis=0)).reshape(1)))
#         u_lim = u_lim_vec[kk]
#         if u_check > u_lim: # and ii > (int(Psub*Nsub)+num_retrain):
#             y_nxt = GP
#         else:
#             y_nxt = np.array(Convert(LS1.Scalar_LS1_HF_2D(inpp))).reshape(1)
#             inp_GPtrain = np.concatenate((inp_GPtrain, inp.reshape(1,Ndim)))
#             y_GPtrain = np.concatenate((y_GPtrain, y_nxt.reshape(1)))
#             GP_pred = np.concatenate((GP_pred, (np.array(GP).reshape(1))))
#             ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
#             amp1, len1, var1 = ML.GP_train(amp_init=amp1, len_init=len1, var_init=var1, num_iters = Iters)
#             var_GP = np.concatenate((var_GP, var1.numpy().reshape(1)))
#             subs_info = np.concatenate((subs_info, np.array(0).reshape(1)))
            
#         if (y_nxt)>y1_lim[kk-1]:
#             inp1[ii,:,kk] = inpp
#             y1[ii,kk] = y_nxt
#         else:
#             inp1[ii,:,kk] = inp1[ind_max,:,kk]
#             y1[ii,kk] = y1[ind_max,kk]

############# AK-SS algorithm #############

# def Norm1(X1,X,dim):
#     K = np.zeros((len(X1),dim))
#     for ii in np.arange(0,dim,1):
#         K[:,ii] = np.reshape(((X1[:,ii])-np.mean((X[:,ii])))/(np.std((X[:,ii]))),len(X1))
#     return K

# def Norm2(X1,X):
#     return ((X1)-np.mean((X)))/(np.std((X)))

# # def InvNorm1(X1,X):
# #     return X1 # (X1*np.std(X,axis=0)+np.mean(X,axis=0))

# def InvNorm2(X1,X):
#     return np.exp(X1*np.std((X))+np.mean((X)))


# def Norm3(X1,X):
#     return ((X1)-np.mean((X)))/(np.std((X)))

# def InvNorm3(X1,X):
#     return (X1*np.std((X))+np.mean((X)))

# Ninit_GP = 12
# lhd0 = lhs(2, samples=Ninit_GP, criterion='centermaximin')
# lhd = uniform(loc=-4,scale=8).ppf(lhd0)
# inp_GPtrain = lhd
# y_GPtrain = np.array(Convert(LS1.Scalar_LS1_HF_2D(inp_GPtrain)))
# ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
# amp1, len1, var1 = ML.GP_train(amp_init=1., len_init=1., var_init=1., num_iters = 1000)
# Iters = 300

# Neval_GP = 3000
# inp_GPeval = np.rot90(np.array([norm(loc=0,scale=1).rvs(Neval_GP),norm(loc=0,scale=1).rvs(Neval_GP)]))
# GP_mean = np.zeros(Neval_GP)
# GP_std = np.zeros(Neval_GP)
# U_func = np.zeros(Neval_GP)
# count = 0
# while np.min(U_func)<=2:
#     # for ii in np.arange(0,Neval_GP,1):
#     #     samples0 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = inp_GPeval[ii,:][None,:], num_samples=num_s)
#     #     GP_mean[ii] = np.array(np.mean(np.array(samples0),axis=0))
#     #     GP_std[ii] = np.array(np.std(np.array(samples0),axis=0))
#     samples0 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = Norm1(inp_GPeval,inp_GPtrain,Ndim), num_samples=num_s)
#     GP_mean = np.array(InvNorm3(np.mean(np.array(samples0),axis=0),y_GPtrain))
#     GP_std = np.std(InvNorm3(np.array(samples0),y_GPtrain),axis=0)
#     U_func = np.abs(GP_mean)/GP_std
#     U_func = U_func.reshape(Neval_GP)
#     ind = np.where(U_func==np.min(U_func))
#     count = count + 1
#     print(count)
#     inp_GPtrain = np.concatenate((inp_GPtrain, inp_GPeval[ind,:].reshape(1,2)))
#     y_GPtrain = np.concatenate((y_GPtrain, np.array(Convert(LS1.Scalar_LS1_HF_2D(inp_GPeval[ind,:].reshape(1,2))))))
#     ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
#     amp1, len1, var1 = ML.GP_train(amp_init=1., len_init=1., var_init=1., num_iters = Iters)
#     inp_GPeval = np.delete(inp_GPeval,ind,0)
#     Neval_GP = Neval_GP-1
    
    
# uni = uniform()
# Nsub = 8000
# Psub = 0.1
# Nlim = 3
# y1 = np.zeros((Nsub,Nlim))
# y1_lim = np.zeros(Nlim)
# y1_lim[Nlim-1] = value
# inp1 = np.zeros((Nsub,Ndim,Nlim))
# rv = norm(loc=0,scale=1)
# y_seed = np.zeros(int(Psub*Nsub))

# for ii in np.arange(0,Nsub,1):
#     inp = DR1.StandardNormal_Indep(N=Ndim)
#     print(ii)
#     inpp = inp[None,:]
#     samples0 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim), num_samples=num_s)
#     y1[ii,0] = np.array(InvNorm3(np.mean(np.array(samples0),axis=0),y_GPtrain))
#     inp1[ii,:,0] = inp

# inpp = np.zeros(Ndim)
# count_max = Nsub/(Psub*Nsub)
# count = 100000
# ind_max = 1
# r_sto = np.zeros((Nsub-int(Psub*Nsub),Nlim-1,Ndim))
# # ind_sto = 3

# for kk in np.arange(1,Nlim,1):
#     ind_max = 0
#     ind_sto = -1
#     count = np.inf
#     y1[0:(int(Psub*Nsub)),kk] = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
#     y1_lim[kk-1] = np.min(y1[0:(int(Psub*Nsub)),kk])
#     indices = (-y1[:,kk-1]).argsort()[:(int(Psub*Nsub))]
#     inp1[0:(int(Psub*Nsub)),:,kk] = inp1[indices,:,kk-1]
#     for ii in np.arange((int(Psub*Nsub)),(Nsub),1):
#         print(kk)
#         print(ii)
#         nxt = np.zeros((1,Ndim))
#         if count > count_max:
#             # ind_max = random.randint(0,int(Psub*Nsub)) # ind_sto
#             ind_sto = ind_sto + 1
#             ind_max = ind_sto
#             count = 0
#         else:
#             ind_max = ii-1
            
#         count = count + 1

#         for jj in np.arange(0,Ndim,1):
#             if jj == 0:
#                 rv1 = norm(loc=inp1[ind_max,jj,kk],scale=0.1)
#             else:
#                 rv1 = norm(loc=inp1[ind_max,jj,kk],scale=0.75)
#             # rv1 = norm(loc=(inp1[ind_max,jj,kk]),scale=1.0)
#             prop = (rv1.rvs())
#             r = np.log(norm.pdf(prop)) - np.log(norm.pdf(inp1[ind_max,jj,kk])) # rv.pdf((prop))/rv.pdf((inp1[ii-(int(Psub*Nsub)),jj,kk]))
#             r_sto[ii-(int(Psub*Nsub)),kk-1,jj] = r
#             if r>np.log(uni.rvs()):
#                 nxt[0,jj] = prop
#             else: 
#                 nxt[0,jj] = inp1[ind_max,jj,kk]
#             inpp[jj] = nxt[0,jj]
#         samples0 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = Norm1(inpp[None,:],inp_GPtrain,Ndim), num_samples=num_s)
#         y_nxt = np.array(InvNorm3(np.mean(np.array(samples0),axis=0),y_GPtrain))
#         if y_nxt>y1_lim[kk-1]:
#             inp1[ii,:,kk] = inpp # np.array([nxt[0,0], nxt[0,1], nxt[0,2]])
#             y1[ii,kk] = y_nxt
#         else:
#             inp1[ii,:,kk] = inp1[ind_max,:,kk]
#             y1[ii,kk] = y1[ind_max,kk]

# Pf = 1
# Pi_sto = np.zeros(Nlim)
# cov_sq = 0
# for kk in np.arange(0,Nlim,1):
#     Pi = len(np.rot90(np.where(y1[:,kk]>np.min([y1_lim[kk],value]))))/(Nsub)
#     Pf = Pf * Pi
#     Pi_sto[kk] = Pi
#     cov_sq = cov_sq + ((1-Pi)/(Pi*Nsub))
# cov_req = np.sqrt(cov_sq)    
    


# y1_lim[Nlim-1] = value
# Pf = 1
# Pi_sto = np.zeros(Nlim)
# cov_sq = 0
# for kk in np.arange(0,Nlim,1):
#     Pi = len(np.rot90(np.where(y1[:,kk]>np.min([y1_lim[kk],value]))))/(Nsub)
#     Pf = Pf * Pi
#     Pi_sto[kk] = Pi
#     cov_sq = cov_sq + ((1-Pi)/(Pi*Nsub))
# cov_req = np.sqrt(cov_sq)

# plt.plot(Out_data[0:2499]*1e-6,label='Subset 0')
# plt.plot(Out_data[2500:4999]*1e-6,label='Subset 1')
# plt.plot(Out_data[4999:7499]*1e-6,label='Subset 2')
# plt.plot(Out_data[7499:9999]*1e-6,label='Subset 3')
# # plt.plot(y1[1000:2000,4],label='Subset 4')
# plt.xlabel('Iteration')
# plt.ylabel('Stress-strength (MPa)')
# plt.legend(loc='lower left')