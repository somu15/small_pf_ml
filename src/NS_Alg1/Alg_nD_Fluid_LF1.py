
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 10:09:55 2021

@author: dhulls
"""

from os import sys
import os
os.chdir('/Users/dhulls/projects/Small Pf/Small_Pf_code/src/NS_Alg1')
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

Ndim = 6
value = 0.85 # 600.0

LS1 = LSF()
DR1 = DR()
num_s = 500
P = np.array([0.045,0.5,1,1,1,1])

## Training GP

def Norm1(X1,X,dim):
    K = np.zeros((len(X1),dim))
    for ii in np.arange(0,dim,1):
        # K[:,ii] = np.reshape(((X1[:,ii])-np.mean((X[:,ii])))/(np.std((X[:,ii]))),len(X1))
        K[:,ii] = X1[:,ii]/X[ii]
    return K

# def Norm2(X1,X):
#     return ((X1)-np.mean((X)))/(np.std((X)))
#
# def InvNorm2(X1,X):
#     return np.exp(X1*np.std((X))+np.mean((X)))


def Norm3(X1,X):
    # return ((X1)-np.mean((X)))/(np.std((X)))
    return X1

def InvNorm3(X1,X):
    # return (X1*np.std((X))+np.mean((X)))
    return X1

## Train the GP diff model

Iters = 400
Ninit_GP = 30
lhd = DR1.FluidLHS(Nsamps=Ninit_GP)
y_LF_GP = np.empty(1, dtype = float)
y_HF_GP = np.empty(1, dtype = float)
inp_GPtrain = lhd
y_LF_GP = LS1.Fluid_S1(inp_GPtrain)
y_HF_GP = LS1.Fluid_NS1(inp_GPtrain)
y_GPtrain = y_HF_GP - y_LF_GP
ML = ML_TF(obs_ind = Norm1(inp_GPtrain,P, Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
amp1, len1 = ML.GP_train(amp_init=1., len_init=1., num_iters = 1000)

## Subset simultion with HF-LF and GP

uni = uniform()
Nsub = 5000
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
Indicator = np.ones((Nsub,Nlim))
counter = 1
# file1 = open('/home/dhullaks/projects/Small_Pf_code/src/NS_Alg1/Results.csv','w')
# file1.writelines("0,0,0\n")
# file1.close()

for ii in np.arange(0,Nsub,1):
    inp = DR1.FluidRandom()
    inpp = inp[None,:]
    LF = LS1.Fluid_S1(inpp)
    inp1[ii,:,0] = inp
    # samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim), num_samples=num_s)
    # GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
    # GP_diff = ML.GP_predict_mean(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim)).reshape(1)
    # additive = 0.0
    # u_check = (np.abs(LF + GP_diff-additive))/ML.GP_predict_std(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim)).reshape(1)
    samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inpp,P,Ndim), num_samples=num_s)
    GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
    additive = value
    u_check = (np.abs(LF + GP_diff - additive))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
    u_GP[ii,0] = u_check
    # if ii > 9:
    #     additive = np.percentile(y1[1:ii,0],90)
    # additive = 0.0
    # u_check = (np.abs(LF + GP_diff - additive))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
    # u_GP = np.concatenate((u_GP, np.array(u_check).reshape(1)))

    u_lim = u_lim_vec[0]
    print(ii)
    if u_check > u_lim:
        y1[ii,0] = LF + GP_diff
    else:
        y1[ii,0] = np.array((LS1.Fluid_NS1(inpp))).reshape(1)
        inp_GPtrain = np.concatenate((inp_GPtrain, inpp.reshape(1,Ndim)))
        y_LF_GP = np.concatenate((y_LF_GP, np.array(LF).reshape(1)))
        y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
        y_GPtrain = np.concatenate((y_GPtrain, (y1[ii,0].reshape(1)-LF)))
        LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
        GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
        # ML = ML_TF(obs_ind = Norm1(inp_GPtrain,inp_GPtrain,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
        # amp1, len1 = ML.GP_train(amp_init=amp1, len_init=len1, num_iters = Iters)
        ML = ML_TF(obs_ind = Norm1(inp_GPtrain,P,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
        amp1, len1 = ML.GP_train(amp_init=amp1, len_init=len1, num_iters = Iters)
        subs_info[ii,0] = 1.0
    # file1 = open('/home/dhullaks/projects/Small_Pf_code/src/NS_Alg1/Results.csv','r')
    # Lines = file1.readlines()
    # Lines = np.concatenate((Lines,np.array(str(counter)+","+str(y1[ii,0])+","+str(subs_info[ii,0])+"\n").reshape(1)))
    # file1 = open('/home/dhullaks/projects/Small_Pf_code/src/NS_Alg1/Results.csv','w')
    # file1.writelines(Lines)
    # file1.close()
    # counter = counter + 1

#std_GPdiff = np.delete(std_GPdiff, 0)
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

# np.log(np.abs(inp1[0:1500,:,0])).reshape(1500,6).std(0)

prop_std_req =np.array([0.375,0.216,0.1875,0.1875,0.1875,0.1875])

for kk in np.arange(1,Nsub,1):
    count = np.inf
    ind_max = 0
    ind_sto = -1
    seeds_outs = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
    y1_lim[kk-1] = np.min(seeds_outs)
    k = (y1[:,kk-1]).argsort()
    indices = k[int((1-Psub)*Nsub):(len(y1))]
    seeds = inp1[indices,:,kk-1]

    for ii in np.arange(1579,(Nsub),1):
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
            #    rv1 = uniform(loc=((inp1[ind_max,jj,kk])-prop_std_req[jj]),scale=(2*prop_std_req[jj]))
            #    prop = (rv1.rvs())
            # else:
            #    rv1 = uniform(loc=(np.log(inp1[ind_max,jj,kk])-prop_std_req[jj]),scale=(2*prop_std_req[jj]))
            #    prop = np.exp(rv1.rvs())
            r = np.log(DR1.FluidPDF(rv_req=prop, index=jj)) - np.log(DR1.FluidPDF(rv_req=(markov_seed[jj]),index=jj)) # np.log(rv.pdf((prop)))-np.log(rv.pdf((inp1[ind_max,jj,kk])))
            if r>np.log(uni.rvs()):
                nxt[0,jj] = prop
            else:
                nxt[0,jj] = markov_seed[jj]
            inpp[0,jj] = nxt[0,jj]
        LF = LS1.Fluid_S1(inpp)
        # samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim), num_samples=num_s)
        # GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
        # GP_diff = ML.GP_predict_mean(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim)).reshape(1)
        # additive = y1_lim[kk-1]
        # u_check = (np.abs(LF + GP_diff-additive))/ML.GP_predict_std(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim)).reshape(1)
        samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inpp,P,Ndim), num_samples=num_s)
        GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
        additive = y1_lim[kk-1]
        u_check = (np.abs(LF + GP_diff - additive))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)

        u_GP[ii,kk] = u_check
        u_lim = u_lim_vec[kk]
        if kk == (Nlim-1):
            additive = value
            u_check1 = (np.abs(LF + GP_diff-additive))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
            u_req[ii] = u_check1

        if u_check > u_lim and u_check1 >= u_lim:
            y_nxt = LF + GP_diff
        else:
            y_nxt = np.array((LS1.Fluid_NS1(inpp))).reshape(1)
            inp_GPtrain = np.concatenate((inp_GPtrain, inpp.reshape(1,Ndim)))
            y_LF_GP = np.concatenate((y_LF_GP, np.array(LF).reshape(1)))
            y_HF_GP = np.concatenate((y_HF_GP, y_nxt.reshape(1)))
            y_GPtrain = np.concatenate((y_GPtrain, (y_nxt.reshape(1)-LF)))
            LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
            GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
            ML = ML_TF(obs_ind = Norm1(inp_GPtrain,P,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
            amp1, len1 = ML.GP_train(amp_init=amp1, len_init=len1, num_iters = Iters)
            subs_info[ii,kk] = 1.0

        if (y_nxt)>y1_lim[kk-1]:
            inp1[ii,:,kk] = inpp
            y1[ii,kk] = y_nxt
        else:
            inp1[ii,:,kk] = markov_seed
            y1[ii,kk] = markov_out
            Indicator[ii,kk] = 0.0
        # file1 = open('/home/dhullaks/projects/Small_Pf_code/src/NS_Alg1/Results.csv','r')
        # Lines = file1.readlines()
        # Lines = np.concatenate((Lines,np.array(str(counter)+","+str(y1[ii,0])+","+str(subs_info[ii,0])+"\n").reshape(1)))
        # file1 = open('/home/dhullaks/projects/Small_Pf_code/src/NS_Alg1/Results.csv','w')
        # file1.writelines(Lines)
        # file1.close()
        # counter = counter + 1

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
#             rv1 = norm(loc=norm().ppf(DR1.FluidCDF(inp1[ind_max,jj,kk],jj)),scale=1.0)
#             prop = DR1.FluidPPF(rv1.cdf(rv1.rvs()),jj)
#             r = np.log(norm().pdf(norm().ppf(DR1.FluidCDF(prop,jj)))) - np.log(norm().pdf(norm().ppf(DR1.FluidCDF(inp1[ind_max,jj,kk],jj)))) # np.log(DR1.FluidPDF(rv_req=prop, index=jj)) - np.log(DR1.FluidPDF(rv_req=(inp1[ind_max,jj,kk]),index=jj)) # np.log(rv.pdf((prop)))-np.log(rv.pdf((inp1[ind_max,jj,kk])))
#             if r>np.log(uni.rvs()):
#                 nxt[0,jj] = prop
#             else:
#                 nxt[0,jj] = inp1[ind_max,jj,kk]
#             inpp[0,jj] = nxt[0,jj]
#         LF = LS1.Fluid_S1(inpp)
#         samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, observation_noise_variance_var=var1, pred_ind = Norm1(inpp,inp_GPtrain,Ndim), num_samples=num_s)
#         GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
#         if ii > 9: # and kk < (Nlim-1):
#             additive = np.percentile(y1[1:ii,kk],90)
#         else:
#             additive = value
#         u_check = (np.abs(LF + GP_diff - additive))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
#         u_GP = np.concatenate((u_GP, np.array(u_check).reshape(1)))
#         std_GPdiff = np.concatenate((std_GPdiff, np.array(np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)).reshape(1)))
#         u_lim = u_lim_vec[kk]
#         if u_check > u_lim:
#             y_nxt = LF + GP_diff
#         else:
#             y_nxt = np.array((LS1.Fluid_NS1(inpp))).reshape(1)
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

#value = 0.0
#y1_lim[Nlim-1] = value

Pf = 1
Pi_sto = np.zeros(Nlim)
cov_sq = 0
for kk in np.arange(0,Nlim,1):
    Pi = len(np.rot90(np.where(y1[:,kk]>np.min([y1_lim[kk],value]))))/(Nsub)
    Pf = Pf * Pi
    Pi_sto[kk] = Pi
    cov_sq = cov_sq + ((1-Pi)/(Pi*Nsub))
cov_req = np.sqrt(cov_sq)

# filename = 'Alg_Run1_GP.pickle'
# os.chdir('/home/dhullaks/projects/Small_Pf_code/src/NS_Alg1')
# with open(filename, 'wb') as f:
#     pickle.dump(y1, f)
#     pickle.dump(y1_lim, f)
#     pickle.dump(Pf, f)
#     pickle.dump(cov_req, f)
#     pickle.dump(Nlim, f)
#     pickle.dump(Nsub, f)
#     pickle.dump(Pi_sto, f)
#     pickle.dump(u_GP, f)
#     pickle.dump(subs_info, f)
#     pickle.dump(y_GPtrain, f)
#     pickle.dump(y_HF_GP, f)
#     pickle.dump(y_LF_GP, f)
#     pickle.dump(Indicator, f)
