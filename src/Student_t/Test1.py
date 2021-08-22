#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 14:31:17 2021

@author: dhulls
"""

from os import sys
import os
import pathlib
import numpy as np
import random
import csv
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import rayleigh
from scipy.stats import uniform
from scipy.stats import cauchy
import matplotlib.pyplot as plt
import pickle

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()
from pyDOE import *

def sinusoid(x):
  return np.sin(3 * np.pi * x[..., 0])

def generate_1d_data(num_training_points, observation_noise_variance):
  """Generate noisy sinusoidal observations at a random set of points.

  Returns:
     observation_index_points, observations
  """
  index_points_ = np.random.uniform(-1., 1., (num_training_points, 1))
  index_points_ = index_points_.astype(np.float64)
  # y = f(x) + noise
  observations_ = (sinusoid(index_points_) +
                   np.random.normal(loc=0,
                                    scale=np.sqrt(observation_noise_variance),
                                    size=(num_training_points)))
  return index_points_, observations_

tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels

num_points = 100
# Index points should be a collection (100, here) of feature vectors. In this
# example, we're using 1-d vectors, so we just need to reshape the output from
# np.linspace, to give a shape of (100, 1).
index_points = np.expand_dims(np.linspace(-1., 1., num_points), -1)

# Define a kernel with default parameters.
kernel = psd_kernels.ExponentiatedQuadratic()

tp = tfd.StudentTProcess(3., kernel, index_points)

samples = tp.sample(10)
# ==> 10 independently drawn, joint samples at `index_points`

noisy_tp = tfd.StudentTProcess(
    df=3.,
    kernel=kernel,
    index_points=index_points)
noisy_samples = noisy_tp.sample(10)
