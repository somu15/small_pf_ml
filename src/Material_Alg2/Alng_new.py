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
value = 400.0

LS1 = LSF()
DR1 = DR()
num_s = 500

P = np.array([200,300,0.25,0.3,135,0.15,0.15,0.15])

def Norm1(X1,X,dim):
    K = np.zeros((len(X1),dim))
    for ii in np.arange(0,dim,1):
        K[:,ii] = X1[:,ii]/X[ii]
    return K

def Norm3(X1,X):
    return (X1/400)

def InvNorm3(X1,X):
    return (X1*400)

Iters = 400
Ninit_GP = 2800
lhd, lhd0 = DR1.MaterialRandom(N=Ninit_GP)
y_LF_GP = np.empty(1, dtype = float)
y_HF_GP = np.empty(1, dtype = float)
inp_GPtrain = lhd
inp_GPtrain0 = lhd0
y_LF_GP = LS1.Material_LF1(inp_GPtrain0)
y_HF_GP = LS1.Material_HF1(inp_GPtrain)
y_GPtrain = y_HF_GP - y_LF_GP
ML = ML_TF(obs_ind = Norm1(inp_GPtrain,P, Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
amp1, len1 = ML.GP_train(amp_init=1., len_init=1., num_iters = 1000)
amp_sto = np.array(amp1).reshape(1)
len_sto = np.array(len1).reshape(1)

uni = uniform()
Nsub = 15000
Psub = 0.1
Nlim = 4
y1 = np.zeros((Nsub,Nlim))
ylf = np.zeros((Nsub,Nlim))
y1_lim = np.zeros(Nlim)
y1_lim[Nlim-1] = value
inp1 = np.zeros((Nsub,Ndim,Nlim))
inp0 = np.zeros((Nsub,Ndim0,Nlim))
rv = norm(loc=0,scale=1)
u_lim_vec = np.array([2,2,2,2,2,2,2,2,2])

u_GP = np.zeros((Nsub,Nlim))
subs_info = np.zeros((Nsub,Nlim))
LF_plus_GP = np.empty(1, dtype = float)
GP_pred = np.empty(1, dtype = float)
additive = value
Indicator = np.ones((Nsub,Nlim))
failed_solve = np.zeros((Nsub,Nlim))

for ii in np.arange(0,Nsub,1):
    inp_HF, inp = DR1.MaterialRandom()
    inp_HF = inp_HF.reshape(Ndim)
    inp = inp.reshape(Ndim0)
    inpp = inp[None,:]
    LF = LS1.Material_LF1(inpp)
    ylf[ii,0] = LF
    inp0[ii,:,0] = inp
    inp1[ii,:,0] = inp_HF[None,:]
    if ii <Ninit_GP:
        additive = np.percentile(y_HF_GP,90)
    else:
        additive = np.percentile(y1[0:ii,0],90)
    samples1 = ML.GP_predict(amplitude_var = amp1, length_scale_var=len1, pred_ind = Norm1(inp_HF[None,:],P,Ndim), num_samples=num_s)
    GP_diff = InvNorm3(np.mean(np.array(samples1),axis=0),y_GPtrain)
    u_check = (np.abs(LF + GP_diff - additive))/np.std(InvNorm3(np.array(samples1),y_GPtrain),axis=0)
    u_GP[ii,0] = u_check
    u_lim = u_lim_vec[0]
    if LF < 0.1:
        u_check = 0
    print(ii)
    if u_check > u_lim:
        y1[ii,0] = LF + GP_diff
    else:
        y1[ii,0] = np.array((LS1.Material_HF1(inp_HF[None,:]))).reshape(1)
        if y1[ii,0] > 0.1:
            inp_GPtrain = np.concatenate((inp_GPtrain, inp_HF[None,:].reshape(1,Ndim)))
            y_LF_GP = np.concatenate((y_LF_GP, np.array(LF).reshape(1)))
            y_HF_GP = np.concatenate((y_HF_GP, y1[ii,0].reshape(1)))
            y_GPtrain = np.concatenate((y_GPtrain, (y1[ii,0].reshape(1)-LF)))
            LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
            GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
            ML = ML_TF(obs_ind = Norm1(inp_GPtrain,P,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
            amp1, len1 = ML.GP_train(amp_init=amp1, len_init=len1, num_iters = Iters)
            amp_sto = np.concatenate((amp_sto, np.array(amp1).reshape(1)))
            len_sto = np.concatenate((len_sto, np.array(len1).reshape(1)))
            subs_info[ii,0] = 1.0
        else:
            y1[ii,0] = np.min(y1[:,0])

LF_plus_GP = np.delete(LF_plus_GP, 0)
GP_pred = np.delete(GP_pred, 0)

inpp = np.zeros((1,Ndim))
count_max = int(Nsub/(Psub*Nsub))-1
seeds_outs = np.zeros(int(Psub*Nsub))
seeds = np.zeros((int(Psub*Nsub),Ndim))
seeds0 = np.zeros((int(Psub*Nsub),Ndim0))
markov_seed = np.zeros(Ndim)
markov_seed0 = np.zeros(Ndim0)
markov_out = 0.0
u_req = np.zeros(Nsub)
u_check1 = 10.0
std_prop = np.zeros(Ndim)

prop_std = np.array([0.75,0.75,0.75,0.75,0.75,1.0,1.0,1.0])
factor1 = np.array([2.0,2.0,1.5,1.0])
ind_LF = [0,2,5,6,7]
inpp = np.zeros((1,Ndim))
LF_prev = 0

for kk in np.arange(1,Nlim,1):

    count = np.inf
    ind_max = 0
    ind_sto = -1
    seeds_outs = np.sort(y1[:,kk-1])[int((1-Psub)*Nsub):(len(y1))]
    y1_lim[kk-1] = np.min(seeds_outs)
    k = (y1[:,kk-1]).argsort()
    indices = k[int((1-Psub)*Nsub):(len(y1))]
    seeds = inp1[indices,:,kk-1]
    seeds0 = inp0[indices,:,kk-1]
    std_prop = np.log((seeds)).std(0)
    tmp = np.zeros((len(seeds_outs),1+Ndim+Ndim0))
    tmp[:,0] = seeds_outs
    tmp[:,1:9] = seeds
    tmp[:,9:14] = seeds0
    np.random.shuffle(tmp)
    seeds_outs = tmp[:,0]
    seeds = tmp[:,1:9]
    seeds0 = tmp[:,9:14]

    for ii in np.arange(14998,(Nsub),1):
        nxt = np.zeros((1,Ndim))

        if count > count_max:
            ind_sto = ind_sto + 1
            count = 0
            markov_seed = seeds[ind_sto,:]
            markov_seed0 = seeds0[ind_sto,:]
            markov_out = seeds_outs[ind_sto]
        else:
            markov_seed = inp1[ii-1,:,kk]
            markov_seed0 = inp0[ii-1,:,kk]
            markov_out = y1[ii-1,kk]

        count = count + 1

        for jj in np.arange(0,Ndim,1):
            rv1 = norm(loc=np.log(markov_seed[jj]),scale=std_prop[jj]) # 0.75
            prop = np.exp(rv1.rvs())

            r = np.log(DR1.MaterialPDF(rv_req=prop, index=jj, LF=0)) - np.log(DR1.MaterialPDF(rv_req=(markov_seed[jj]),index=jj,LF=0))
            if r>np.log(uni.rvs()):
                nxt[0,jj] = prop
            else:
                nxt[0,jj] = markov_seed[jj]
            inpp[0,jj] = nxt[0,jj]

        LF = LS1.Material_LF1(inpp[0,ind_LF].reshape(1,Ndim0))
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

        if LF<0.1:
            u_check = 0

        print(ii)
        print(kk)
        if u_check > u_lim and ii>=Ninit_GP:
            y_nxt = LF + GP_diff
        else:
            y_nxt = np.array((LS1.Material_HF1(inpp))).reshape(1)
            if y_nxt > 0.1:
                inp_GPtrain = np.concatenate((inp_GPtrain, inpp.reshape(1,Ndim)))
                y_LF_GP = np.concatenate((y_LF_GP, np.array(LF).reshape(1)))
                y_HF_GP = np.concatenate((y_HF_GP, y_nxt.reshape(1)))
                y_GPtrain = np.concatenate((y_GPtrain, (y_nxt.reshape(1)-LF)))
                LF_plus_GP = np.concatenate((LF_plus_GP, (LF + np.array(GP_diff).reshape(1))))
                GP_pred = np.concatenate((GP_pred, (np.array(GP_diff).reshape(1))))
                ML = ML_TF(obs_ind = Norm1(inp_GPtrain,P,Ndim), obs = Norm3(y_GPtrain,y_GPtrain))
                amp1, len1 = ML.GP_train(amp_init=amp1, len_init=len1, num_iters = Iters)
                amp_sto = np.concatenate((amp_sto, np.array(amp1).reshape(1)))
                len_sto = np.concatenate((len_sto, np.array(len1).reshape(1)))
                subs_info[ii,kk] = 1.0

            else:
                y_nxt = np.min(y1[:,kk])

        if (y_nxt)>y1_lim[kk-1]:
            inp1[ii,:,kk] = inpp
            inp0[ii,:,kk] = inpp[0,ind_LF]
            y1[ii,kk] = y_nxt
            LF_prev = LF
        else:
            inp1[ii,:,kk] = markov_seed
            inp0[ii,:,kk] = markov_seed0
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
