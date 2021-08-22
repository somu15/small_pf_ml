#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:30:39 2020

@author: dhulls
"""

import numpy as np
import tensorflow.compat.v2 as tf
# tf.compat.v1.disable_eager_execution()
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()
from scipy.stats import norm
from scipy.stats import multivariate_normal

from tensorflow import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
print(tf.__version__)
# import tensorflow_docs as tfdocs
# import tensorflow_docs.plots
# import tensorflow_docs.modeling

class ML_TFvar:
    
    def __init__(self, obs_ind=None, obs=None): #, amp_init=None, len_init=None, var_init=None, num_iters=None):
        self.obs_ind = obs_ind
        self.obs = obs
        
    def GP_train(self, amp_init=None, len_init=None, var_init=None, num_iters=None): # Gaussian Process Regression training
        
        def build_gp(amplitude, length_scale, observation_noise_variance):

          kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

          return tfd.GaussianProcess(
              kernel=kernel,
              index_points=self.obs_ind,
              observation_noise_variance=observation_noise_variance) # ,jitter=1e-03
      
        gp_joint_model = tfd.JointDistributionNamed({
            'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
            'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
            'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
            'observations': build_gp,
        })
        
        amplitude_ = tf.Variable(initial_value=amp_init, name='amplitude_', dtype=np.float64,constraint=lambda z: tf.clip_by_value(z, 1e-4, 10000)) # lambda z: tf.clip_by_value(z, 0, 10000)
        length_scale_ = tf.Variable(initial_value=len_init, name='length_scale_', dtype=np.float64,constraint=lambda z: tf.clip_by_value(z, 1e-4, 10000))
        observation_noise_variance_ = tf.Variable(initial_value=var_init,
                                                  name='observation_noise_variance_',
                                                  dtype=np.float64,constraint=lambda z: tf.clip_by_value(z, 1e-4, 10000))
        
        @tf.function(autograph=False, experimental_compile=False)
        def target_log_prob(amplitude, length_scale, observation_noise_variance):
          return gp_joint_model.log_prob({
              'amplitude': amplitude,
              'length_scale': length_scale,
              'observation_noise_variance': observation_noise_variance,
              'observations': self.obs
          })
        
        optimizer = tf.optimizers.Adam(learning_rate=.01)
        
        for i in range(num_iters):
            with tf.GradientTape() as tape:
                loss = -target_log_prob(amplitude_, length_scale_,
                                observation_noise_variance_)
            grads = tape.gradient(loss, [amplitude_, length_scale_, observation_noise_variance_])
            optimizer.apply_gradients(zip(grads, [amplitude_, length_scale_, observation_noise_variance_]))
          
        print('Trained parameters:')
        print('amplitude: {}'.format(amplitude_.numpy()))
        print('length_scale: {}'.format(length_scale_.numpy()))
        print('observation_noise_variance: {}'.format(observation_noise_variance_.numpy()))
        
        return amplitude_, length_scale_, observation_noise_variance_
    
    def GP_predict(self, amplitude_var=None, length_scale_var=None, observation_noise_variance_var=None, pred_ind=None, num_samples=None): # Gaussian Process Regression prediction
        
        # Gaussian Process Regression prediction
    
        optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)
        gprm = tfd.GaussianProcessRegressionModel(
            kernel=optimized_kernel,
            index_points=pred_ind,
            observation_index_points=self.obs_ind,
            observations=self.obs,
            observation_noise_variance=observation_noise_variance_var,
            predictive_noise_variance=0.)
        
        return gprm.sample(num_samples)
    
    def GP_predict_mean(self, amplitude_var=None, length_scale_var=None, observation_noise_variance_var=None, pred_ind=None): # Gaussian Process Regression prediction
        
        # Gaussian Process Regression prediction
    
        optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)
        gprm = tfd.GaussianProcessRegressionModel(
            kernel=optimized_kernel,
            index_points=pred_ind,
            observation_index_points=self.obs_ind,
            observations=self.obs,
            observation_noise_variance=observation_noise_variance_var,
            predictive_noise_variance=0.)
        
        return np.array(gprm.mean())


    def DNN_train(self, dim=None, seed=None, neurons1=None, neurons2=None, learning_rate=None, epochs=None):

        np.random.seed(seed)
        random.set_seed(seed)

        def norm(X1,dim):
            K = np.zeros((len(X1),dim))
            for ii in np.arange(0,dim,1):
                K[:,ii] = np.reshape((X1[:,ii]-np.mean(X1[:,ii]))/(np.std(X1[:,ii])),len(X1))
            return K

        normed_obs_ind = norm(self.obs_ind, dim)
        normed_obs = norm(self.obs.reshape(len(self.obs),1), 1)

        def build_model():
          model = keras.Sequential([
            layers.Dense(neurons1, activation='softmax', bias_initializer='zeros'), # input_shape=[len(train_dataset.keys())],
            layers.Dense(neurons2, activation='softmax',bias_initializer='zeros'),
            layers.Dense(1,bias_initializer='zeros')
          ])

          # optimizer = tf.keras.optimizers.RMSprop(0.001)
          optimizer = tf.keras.optimizers.RMSprop(learning_rate)

          model.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mae', 'mse'])
          return model

        model = build_model()

        # Inspect the model

        model.fit(
          normed_obs_ind, normed_obs,
          epochs=epochs, validation_split = 0.0,
          shuffle = False)

        model.summary()


        return model

    def DNN_pred(self, ref_ind=None, ref_obs=None, model=None, dim=None, pred_ind=None):

        def norm(X1,dim,ref):
            K = np.zeros((len(X1),dim))
            for ii in np.arange(0,dim,1):
                K[:,ii] = np.reshape((X1[:,ii]-np.mean(ref[:,ii]))/(np.std(ref[:,ii])),len(X1))
            return K

        def invnorm(X1, ref):
            # K = np.zeros((len(X1),dim))
            # for ii in np.arange(0,dim,1):
            #     K[:,ii] = (X1[:,ii] * (np.std(X1[:,ii])),len(X1))+np.mean(X1[:,ii])
            return (X1[:,0] * (np.std(ref[:,0])),len(X1))+np.mean(ref[:,0])

        normed_pred_ind = norm(pred_ind, dim, ref_ind)

        normed_pred = model.predict(normed_pred_ind)

        return invnorm(normed_pred, ref_obs.reshape(len(ref_obs),1))
