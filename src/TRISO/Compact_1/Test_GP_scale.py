#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 21:23:28 2021

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

def generate_1d_data(num_training_points, observation_noise_variance):
  
  index_points_ = np.random.uniform(-2., 2., (num_training_points, 2))
  index_points_ = index_points_.astype(np.float64)
  LS1 = LSF()
  observations_ = (LS1.Scalar_LS1(Input_vec=index_points_) +
                   np.random.normal(loc=0,
                                    scale=np.sqrt(observation_noise_variance),
                                    size=(num_training_points)))
  return index_points_, observations_

NUM_TRAINING_POINTS = 150
obs_ind_, obs_ = generate_1d_data(
    num_training_points=NUM_TRAINING_POINTS,
    observation_noise_variance=.01)

xv = np.linspace(-2., 2., 75)
yv = np.linspace(-2., 2., 75)
predictive_index_points_ = np.zeros((len(xv)*len(yv),2))
count = 0
for i in np.arange(0,len(xv),1):
    for j in np.arange(0,len(yv),1):
        predictive_index_points_[count,0] = xv[i]
        predictive_index_points_[count,1] = yv[j]
        count = count + 1

ML = ML_TF(obs_ind = obs_ind_, obs = obs_)

amp1, len1 = ML.GP_train_kernel(num_iters = 1000)

amp2, len2 = ML.GP_train(amp_init=1., len_init=1., num_iters = 1000)

num_s = 50
samples1 = ML.GP_predict_kernel(amplitude_var = amp1, length_scale_var=len1, pred_ind = predictive_index_points_, num_samples=num_s)
samples2 = ML.GP_predict(amplitude_var = amp2, length_scale_var=len2, pred_ind = predictive_index_points_, num_samples=num_s)

X1,Y1 = np.meshgrid(xv,yv)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X1,Y1, np.mean(np.array(samples2),axis=0).reshape(75,75), alpha=0.1,
                      linewidth=0, antialiased=False,
          label='Posterior Sample' if i == 0 else None)
# ax.set_zlim(0.11, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
# ax.scatter(obs_ind_[:, 0], obs_ind_[:, 1], obs_, zdir='z', s=20)
# plt.legend(loc='upper right')
# plt.show()