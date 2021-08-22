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

# num_samps = 5000
# rv1 = norm(np.log(8.9e-4),0.2) # Viscosity
# rv2 = norm(np.log(1000),0.2) # Density
# rv3 = norm(np.log(5000),1.0) # Pressure

# inputs = np.zeros((num_samps,6))
# inputs[:,0] = np.exp(rv1.rvs(num_samps))
# inputs[:,1] = np.exp(rv2.rvs(num_samps))
# inputs[:,2] = np.exp(rv3.rvs(num_samps))
# inputs[:,3] = np.exp(rv3.rvs(num_samps))
# inputs[:,4] = np.exp(rv3.rvs(num_samps))
# inputs[:,5] = np.exp(rv3.rvs(num_samps))

# rv4 = norm(np.log(1500**2),1.0) # Co_sq
# inputs_D = np.zeros((num_samps,5))
# inputs_D[:,0] = np.exp(rv4.rvs(num_samps))
# inputs_D[:,1:4] = inputs[:,2:5]


# NS_Pressures = LS1.Fluid_NS(inputs)

# S_Pressures = LS1.Fluid_S(inputs)

# D_Pressures = LS1.Fluid_D(inputs_D)

Ndim = 6
value = 0.85 # 600.0

LS1 = LSF()
DR1 = DR()
num_s = 500

## Monte Carlo simulations

# Nsims = int(5000)
# inps = np.zeros((Nsims,Ndim))
# yns = np.zeros(Nsims)
# ys = np.zeros(Nsims)
# LS1 = LSF()
# DR1 = DR()

# for ii in np.arange(0,Nsims,1):
#     inp = (DR1.FluidRandom())
#     inpp = inp[None,:]
#     inps[ii,:] = inp
#     yns[ii] = np.array(LS1.Fluid_NS(inpp))
#     ys[ii] = np.array(LS1.Fluid_S(inpp))
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
# counter = 1
# file1 = open('/home/dhullaks/projects/Small_Pf_code/src/NS_SS2/Results.csv','w')
# file1.writelines("0,0,0\n")
# file1.close()

for ii in np.arange(0,Nsub,1):
    inp = (DR1.FluidRandom())
    inpp = inp[None,:]
    y1[ii,0] = np.array(LS1.Fluid_NS(inpp))
    inp1[ii,:,0] = inp
    print(ii)
    # file1 = open('/home/dhullaks/projects/Small_Pf_code/src/NS_SS2/Results.csv','r')
    # Lines = file1.readlines()
    # Lines = np.concatenate((Lines,np.array(str(counter)+","+str(y1[ii,0])+","+str(subs_info[ii,0])+"\n").reshape(1)))
    # file1 = open('/home/dhullaks/projects/Small_Pf_code/src/NS_SS2/Results.csv','w')
    # file1.writelines(Lines)
    # file1.close()
    # counter = counter + 1

inpp = np.zeros(Ndim)
count_max = Nsub/(Psub*Nsub)
seeds_outs = np.zeros(int(Psub*Nsub))
seeds = np.zeros((int(Psub*Nsub),Ndim))
markov_seed = np.zeros(Ndim)
markov_out = 0.0
r_sto = np.zeros((Nsub-int(Psub*Nsub),Nlim-1,Ndim))
# ind_sto = 3
prop_std_req =np.array([0.375,0.216,0.1875,0.1875,0.1875,0.1875])

for kk in np.arange(1,Nlim,1):
    count = np.inf
    ind_max = 0
    ind_sto = -1
    seeds_outs = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
    y1_lim[kk-1] = np.min(seeds_outs)
    k = (y1[:,kk-1]).argsort()
    indices = k[int((1-Psub)*Nsub):(len(y1))]
    seeds = inp1[indices,:,kk-1]

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
            rv1 = norm(loc=np.log(markov_seed[jj]),scale=0.7)
            prop = np.exp(rv1.rvs())
            # if jj == 1:
            #     rv1 = uniform(loc=((inp1[ind_max,jj,kk])-prop_std_req[jj]),scale=(2*prop_std_req[jj]))
            #     prop = (rv1.rvs())
            # else:
            #     rv1 = uniform(loc=(np.log(inp1[ind_max,jj,kk])-prop_std_req[jj]),scale=(2*prop_std_req[jj]))
            #     prop = np.exp(rv1.rvs())
            r = np.log(DR1.FluidPDF(rv_req=(prop), index=jj)) - np.log(DR1.FluidPDF(rv_req=(markov_seed[jj]),index=jj)) # rv.pdf((prop))/rv.pdf((inp1[ii-(int(Psub*Nsub)),jj,kk]))
            # r_sto[ii-(int(Psub*Nsub)),kk-1,jj] = r
            if r>np.log(uni.rvs()):
                nxt[0,jj] = prop
            else:
                nxt[0,jj] = markov_seed[jj]
            inpp[jj] = nxt[0,jj]
        y_nxt = np.array(LS1.Fluid_NS(inpp[None,:])).reshape(1)
        if y_nxt>y1_lim[kk-1]:
            inp1[ii,:,kk] = inpp # np.array([nxt[0,0], nxt[0,1], nxt[0,2]])
            y1[ii,kk] = y_nxt
        else:
            inp1[ii,:,kk] = markov_seed
            y1[ii,kk] = markov_out
            Indicator[ii,kk] = 0.0
        # file1 = open('/home/dhullaks/projects/Small_Pf_code/src/NS_SS2/Results.csv','r')
        # Lines = file1.readlines()
        # Lines = np.concatenate((Lines,np.array(str(counter)+","+str(y1[ii,0])+","+str(subs_info[ii,0])+"\n").reshape(1)))
        # file1 = open('/home/dhullaks/projects/Small_Pf_code/src/NS_SS2/Results.csv','w')
        # file1.writelines(Lines)
        # file1.close()
        # counter = counter + 1

# value = 0.0
# y1_lim[Nlim-1] = value
Pf = 1
Pi_sto = np.zeros(Nlim)
cov_sq = 0
for kk in np.arange(0,Nlim,1):
    Pi = len(np.rot90(np.where(y1[:,kk]>np.min([y1_lim[kk],value]))))/(Nsub)
    Pf = Pf * Pi
    Pi_sto[kk] = Pi
    cov_sq = cov_sq + ((1-Pi)/(Pi*Nsub))
cov_req = np.sqrt(cov_sq)

# filename = 'SS_Run2.pickle'
# os.chdir('/home/dhullaks/projects/Small_Pf_code/src/NS_SS2')
# with open(filename, 'wb') as f:
#     pickle.dump(y1, f)
#     pickle.dump(y1_lim, f)
#     pickle.dump(Pf, f)
#     pickle.dump(cov_req, f)
#     pickle.dump(Nlim, f)
#     pickle.dump(Nsub, f)
#     pickle.dump(Pi_sto, f)
#     pickle.dump(Indicator, f)

# # Pf = 0.0003651 [cov_req = 0.0437; 5 subsets with 15000 sims per subset]

## Training GP

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
# lhd = DR1.FluidLHS(Nsamps=Ninit_GP)
# y_LF_GP = np.empty(1, dtype = float)
# y_HF_GP = np.empty(1, dtype = float)
# inp_GPtrain = lhd
# y_LF_GP = LS1.Fluid_S(inp_GPtrain)
# y_HF_GP = LS1.Fluid_NS(inp_GPtrain)
# y_GPtrain = y_HF_GP - y_LF_GP
# ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain, Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
# amp1, len1, var1 = ML.GP_train(amp_init=1., len_init=1., var_init=1., num_iters = 1000)

# ## Subset simultion with HF-LF and GP

# uni = uniform()
# Nsub = 2000
# Psub = 0.1
# Nlim = 5
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

# for ii in np.arange(0,Nsub,1):
#     inp = DR1.FluidRandom()
#     inpp = inp[None,:]
#     LF = LS1.Fluid_S(inpp)
#     inp1[ii,:,0] = inp
#     samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim), num_samples=num_s)
#     GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
#     u_check = (np.abs(LF + GP_diff - value))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
#     u_GP = np.concatenate((u_GP, np.array(u_check).reshape(1)))
#     std_GPdiff = np.concatenate((std_GPdiff, np.array(np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)).reshape(1)))
#     u_lim = u_lim_vec[0]
#     if u_check > u_lim:
#         y1[ii,0] = LF + GP_diff
#     else:
#         y1[ii,0] = np.array((LS1.Fluid_NS(inpp))).reshape(1)
#         inp_GPtrain = np.concatenate((inp_GPtrain, inp.reshape(1,Ndim)))
#         y_LF_GP = np.concatenate((y_LF_GP, np.array(LF).reshape(1)))
#         y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
#         y_GPtrain = np.concatenate((y_GPtrain, (y1[ii,0].reshape(1)-LF)))
#         LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
#         GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
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

# prop_std = np.array([1.0,1.0,1.0,1.0,1.0,1.0])

# for kk in np.arange(1,Nlim,1):
#     count = np.inf
#     ind_max = 0
#     ind_sto = -1
#     y1[0:(int(Psub*Nsub)),kk] = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
#     y1_lim[kk-1] = np.min(y1[0:(int(Psub*Nsub)),kk])
#     indices = (-y1[:,kk-1]).argsort()[:(int(Psub*Nsub))]
#     inp1[0:(int(Psub*Nsub)),:,kk] = inp1[indices,:,kk-1]
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
#             r = np.log(DR1.FluidPDF(rv_req=prop, index=jj)) - np.log(DR1.FluidPDF(rv_req=(inp1[ind_max,jj,kk]),index=jj)) # np.log(rv.pdf((prop)))-np.log(rv.pdf((inp1[ind_max,jj,kk])))
#             if r>np.log(uni.rvs()):
#                 nxt[0,jj] = prop
#             else:
#                 nxt[0,jj] = inp1[ind_max,jj,kk]
#             inpp[0,jj] = nxt[0,jj]
#         LF = LS1.Fluid_S(inpp)
#         samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim), num_samples=num_s)
#         GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
#         u_check = (np.abs(LF + GP_diff - value))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
#         u_GP = np.concatenate((u_GP, np.array(u_check).reshape(1)))
#         std_GPdiff = np.concatenate((std_GPdiff, np.array(np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)).reshape(1)))
#         u_lim = u_lim_vec[kk]
#         if u_check > u_lim:
#             y_nxt = LF + GP_diff
#         else:
#             y_nxt = np.array((LS1.Fluid_NS(inpp))).reshape(1)
#             inp_GPtrain = np.concatenate((inp_GPtrain, inpp.reshape(1,Ndim)))
#             y_LF_GP = np.concatenate((y_LF_GP, np.array(LF).reshape(1)))
#             y_HF_GP = np.concatenate((y_HF_GP, y_nxt.reshape(1)))
#             y_GPtrain = np.concatenate((y_GPtrain, (y_nxt.reshape(1)-LF)))
#             LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
#             GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
#             ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
#             amp1, len1, var1 = ML.GP_train(amp_init=amp1, len_init=len1, var_init=var1, num_iters = Iters)
#             var_GP = np.concatenate((var_GP, var1.numpy().reshape(1)))
#             subs_info = np.concatenate((subs_info, np.array(kk).reshape(1)))

#         if (y_nxt)>y1_lim[kk-1]:
#             inp1[ii,:,kk] = inpp
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
