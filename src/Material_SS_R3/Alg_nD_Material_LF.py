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
value = 130.0

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
Nsub = 2000
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

for kk in np.arange(1,Nlim,1):
    count = np.inf
    ind_max = 0
    ind_sto = -1
    seeds_outs = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
    y1_lim[kk-1] = np.min(seeds_outs)
    k = (y1[:,kk-1]).argsort()
    indices = k[int((1-Psub)*Nsub):(len(y1))]
    seeds = inp1[indices,:,kk-1]

    for ii in np.arange(0,Nsub,1):
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
            rv1 = norm(loc=np.log(markov_seed[jj]),scale=0.75)
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

# filename = 'SS_Run3.pickle'
# os.chdir('/home/dhullaks/projects/Small_Pf_code/src/Material_SS_R3')
# with open(filename, 'wb') as f:
#     pickle.dump(y1, f)
#     pickle.dump(y1_lim, f)
#     pickle.dump(Pf, f)
#     pickle.dump(cov_req, f)
#     pickle.dump(Nlim, f)
#     pickle.dump(Nsub, f)
#     pickle.dump(Pi_sto, f)
#     pickle.dump(Indicator, f)

## Pf =

# ## Training GP

# value = 375

# def Norm1(X1,X,dim):
#     K = np.zeros((len(X1),dim))
#     for ii in np.arange(0,dim,1):
#         K[:,ii] = np.reshape((np.log(X1[:,ii])-np.mean(np.log(X[:,ii])))/(np.std(np.log(X[:,ii]))),len(X1))
#     return K

# def Norm2(X1,X):
#     return (np.log(X1)-np.mean(np.log(X)))/(np.std(np.log(X)))

# # def InvNorm1(X1,X):
# #     return X1 # (X1*np.std(X,axis=0)+np.mean(X,axis=0))

# def InvNorm2(X1,X):
#     return np.exp(X1*np.std(np.log(X))+np.mean(np.log(X)))


# def Norm3(X1,X):
#     return ((X1)-np.mean((X)))/(np.std((X)))

# def InvNorm3(X1,X):
#     return (X1*np.std((X))+np.mean((X)))

# ## Train the GP diff model

# Iters = 300
# Ninit_GP = 12
# lhd, lhd0 = DR1.MaterialLHS(Nsamps=Ninit_GP)
# y_LF_GP = np.empty(1, dtype = float)
# y_HF_GP = np.empty(1, dtype = float)
# inp_GPtrain = lhd
# inp_GPtrain0 = lhd0
# y_LF_GP = LS1.Material_LF(inp_GPtrain0)
# y_HF_GP = LS1.Material_HF(inp_GPtrain)
# y_GPtrain = y_HF_GP - y_LF_GP
# ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain, Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
# amp1, len1, var1 = ML.GP_train(amp_init=1., len_init=1., var_init=0.001, num_iters = 1000)

# ## Subset simultion with HF-LF and GP

# uni = uniform()
# Nsub = 2500
# Psub = 0.1
# Nlim = 5
# y1 = np.zeros((Nsub,Nlim))
# y1_lim = np.zeros(Nlim)
# y1_lim[Nlim-1] = value
# inp1 = np.zeros((Nsub,Ndim,Nlim))
# inp0 = np.zeros((Nsub,Ndim0,Nlim))
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
#     inp_HF, inp = DR1.MaterialRandom()
#     inpp = inp[None,:]
#     # samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, observation_noise_variance_var=var0, pred_ind = Norm1(inpp,inp_LFtrain,Ndim), num_samples=num_s)
#     # LF = np.array(InvNorm2(np.mean(np.array(samples0),axis=0),y_HF_LFtrain))
#     LF = LS1.Material_LF(inpp)
#     inp0[ii,:,0] = inp
#     inp1[ii,:,0] = inp_HF[None,:]
#     samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = Norm1(inp_HF[None,:],inp_GPtrain,Ndim), num_samples=num_s)
#     GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
#     # if ii > 9:
#     #     additive = np.percentile(y1[:,0],90)
#     u_check = (np.abs(LF + GP_diff-additive))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
#     u_GP = np.concatenate((u_GP, np.array(u_check).reshape(1)))
#     std_GPdiff = np.concatenate((std_GPdiff, np.array(np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)).reshape(1)))
#     u_lim = u_lim_vec[0]
#     print(ii)
#     if u_check > u_lim:
#         y1[ii,0] = LF + GP_diff
#     else:
#         y1[ii,0] = np.array((LS1.Material_HF(inp_HF[None,:]))).reshape(1)
#         inp_GPtrain = np.concatenate((inp_GPtrain, inp_HF[None,:].reshape(1,Ndim)))
#         y_LF_GP = np.concatenate((y_LF_GP, np.array(LF).reshape(1)))
#         y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
#         y_GPtrain = np.concatenate((y_GPtrain, (y1[ii,0].reshape(1)-LF)))
#         LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
#         GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
#         # ML = ML_TF(obs_ind = (np.array(inp_GPtrain))[:,:,0], obs = (np.array(y_HF_GP)[:,:,0]-np.array(y_LF_GP)[:,:,0])[:,0])
#         ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
#         amp1, len1, var1 = ML.GP_train(amp_init=amp1, len_init=len1, var_init=var1, num_iters = Iters)
#         var_GP = np.concatenate((var_GP, var1.numpy().reshape(1)))
#         subs_info = np.concatenate((subs_info, np.array(0).reshape(1)))

# u_GP = np.delete(u_GP, 0)
# var_GP = np.delete(var_GP, 0)
# std_GPdiff = np.delete(std_GPdiff, 0)
# subs_info = np.delete(subs_info, 0)
# LF_plus_GP = np.delete(LF_plus_GP, 0)
# GP_pred = np.delete(GP_pred, 0)

# count_max = int(Nsub/(Psub*Nsub))

# prop_std = np.array([1.0,1.0,1.0,1.0,1.0])
# ind_LF = [0,2,5,6,7]
# inpp = np.zeros((1,Ndim))
# for kk in np.arange(1,Nlim,1):
#     count = np.inf
#     ind_max = 0
#     ind_sto = -1
#     y1[0:(int(Psub*Nsub)),kk] = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
#     y1_lim[kk-1] = np.min(y1[0:(int(Psub*Nsub)),kk])
#     indices = (-y1[:,kk-1]).argsort()[:(int(Psub*Nsub))]
#     inp1[0:(int(Psub*Nsub)),:,kk] = inp1[indices,:,kk-1]
#     inp0[0:(int(Psub*Nsub)),:,kk] = inp0[indices,:,kk-1]
#     for ii in np.arange((int(Psub*Nsub)),(Nsub),1):
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
#             rv1 = norm(loc=np.log(inp1[ind_max,jj,kk]),scale=1.0)
#             prop = np.exp(rv1.rvs())
#             r = np.log(DR1.MaterialPDF(rv_req=prop, index=jj, LF=0)) + np.log(rv1.pdf(np.log(inp1[ind_max,jj,kk]))) - np.log(rv1.pdf(np.log(prop))) - np.log(DR1.MaterialPDF(rv_req=(inp1[ind_max,jj,kk]),index=jj,LF=0)) # np.log(rv.pdf((prop)))-np.log(rv.pdf((inp1[ind_max,jj,kk])))
#             if r>np.log(uni.rvs()):
#                 nxt[0,jj] = prop
#             else:
#                 nxt[0,jj] = inp1[ind_max,jj,kk]
#             inpp[0,jj] = nxt[0,jj]
#         # samples0 = ML0.GP_predict(amplitude_var = amp0, length_scale_var=len0, observation_noise_variance_var=var0, pred_ind = Norm1(inpp,inp_LFtrain,Ndim), num_samples=num_s)
#         # LF = np.array(InvNorm2(np.mean(np.array(samples0),axis=0),y_HF_LFtrain))
#         LF = LS1.Material_LF(inpp[0,ind_LF].reshape(1,Ndim0))
#         samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim), num_samples=num_s)
#         GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
#         # if ii > 9 and kk < (Nlim-1):
#         #     additive = np.percentile(y1[:,kk],90)
#         # else:
#         #     additive = value
#         u_check = (np.abs(LF + GP_diff - additive))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
#         u_GP = np.concatenate((u_GP, np.array(u_check).reshape(1)))
#         std_GPdiff = np.concatenate((std_GPdiff, np.array(np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)).reshape(1)))
#         u_lim = u_lim_vec[0]
#         print(ii)
#         if u_check > u_lim: # and ii > (int(Psub*Nsub)+num_retrain):
#             y_nxt = LF + GP_diff
#         else:
#             y_nxt = np.array((LS1.Material_HF(inpp))).reshape(1)
#             inp_GPtrain = np.concatenate((inp_GPtrain, inpp.reshape(1,Ndim)))
#             y_LF_GP = np.concatenate((y_LF_GP, np.array(LF).reshape(1)))
#             y_HF_GP = np.concatenate((y_HF_GP, y_nxt.reshape(1)))
#             y_GPtrain = np.concatenate((y_GPtrain, (y_nxt.reshape(1)-LF)))
#             LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
#             GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
#             # ML = ML_TF(obs_ind = (np.array(inp_GPtrain))[:,:,0], obs = (np.array(y_HF_GP)[:,:,0]-np.array(y_LF_GP)[:,:,0])[:,0])
#             ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
#             amp1, len1, var1 = ML.GP_train(amp_init=amp1, len_init=len1, var_init=var1, num_iters = Iters)
#             var_GP = np.concatenate((var_GP, var1.numpy().reshape(1)))
#             subs_info = np.concatenate((subs_info, np.array(kk).reshape(1)))

#         if (y_nxt)>y1_lim[kk-1]:
#             inp1[ii,:,kk] = inpp
#             inp0[ii,:,kk] = inpp[0,ind_LF]
#             y1[ii,kk] = y_nxt
#         else:
#             inp1[ii,:,kk] = inp1[ind_max,:,kk]
#             inp0[ii,:,kk] = inp0[ind_max,:,kk]
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

# # req = 2.425e-04 (1000 sims per subset and 4 subsets; Num HF = 86; value=800)

# # plt.scatter(y_HF_GP[12:86],LF_plus_GP)
# # plt.xlim([700,900])
