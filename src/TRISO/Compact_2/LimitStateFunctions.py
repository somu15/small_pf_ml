#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 09:55:13 2020

@author: dhulls
"""

import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import gumbel_r, gumbel_l
from scipy.stats import lognorm
from scipy.stats import truncnorm
from TrisoModel import TrisoModel as TM

class LimitStateFunctions:

    # def Triso_2d(self, Input_vec=None):
    #
    #     # Input_vec = [E, v, ux]
    #     #             [0, 1,  2]
    #
    #     siz1 = len(Input_vec[:,0])
    #     out1 = np.zeros(siz1)
    #     TM1 = TM()
    #
    #     for ii in np.arange(0,siz1,1):
    #         out1[ii] = (TM1.Triso_2d(Input_vec[ii,0],Input_vec[ii,1],Input_vec[ii,2],Input_vec[ii,3],Input_vec[ii,4],Input_vec[ii,5]))
    #
    #     return out1

    def Triso_1d(self, Input_vec=None):

        siz1 = len(Input_vec[:,0])
        out1 = np.zeros(siz1)
        TM1 = TM()

        for ii in np.arange(0,siz1,1):
            out1[ii] = (TM1.Triso_1d(Input_vec[ii,0],Input_vec[ii,1],Input_vec[ii,2],Input_vec[ii,3],Input_vec[ii,4],Input_vec[ii,5],Input_vec[ii,6],Input_vec[ii,7])) #

        return out1

    def Triso_1d_norm(self, Input_vec=None):

        siz1 = len(Input_vec[:,0])
        out1 = np.zeros(siz1)
        TM1 = TM()
        Input_vec_norm = np.zeros((siz1,8))
        rv1 = norm(loc=213.35e-6,scale=4.4e-6) # kernel_r
        rv2 = norm(loc=98.9e-6,scale=8.4e-6) # buffer_t
        rv3 = norm(loc=40.4e-6,scale=2.5e-6) # ipyc_t
        rv4 = norm(loc=35.2e-6,scale=1.2e-6) # sic_t
        rv5 = norm(loc=43.4e-6,scale=2.9e-6) # opyc_t
        rv6 = uniform()
        rv_norm = norm(loc=0,scale=1)

        Input_vec_norm[:,0] = rv1.ppf(rv_norm.cdf(Input_vec[:,0]))
        Input_vec_norm[:,1] = rv2.ppf(rv_norm.cdf(Input_vec[:,1]))
        Input_vec_norm[:,2] = rv3.ppf(rv_norm.cdf(Input_vec[:,2]))
        Input_vec_norm[:,3] = rv4.ppf(rv_norm.cdf(Input_vec[:,3]))
        Input_vec_norm[:,4] = rv5.ppf(rv_norm.cdf(Input_vec[:,4]))
        Input_vec_norm[:,5] = rv6.ppf(rv_norm.cdf(Input_vec[:,5]))
        Input_vec_norm[:,6] = rv6.ppf(rv_norm.cdf(Input_vec[:,6]))
        Input_vec_norm[:,7] = rv6.ppf(rv_norm.cdf(Input_vec[:,7]))

        for ii in np.arange(0,siz1,1):
            out1[ii] = (TM1.Triso_1d(Input_vec_norm[ii,0],Input_vec_norm[ii,1],Input_vec_norm[ii,2],Input_vec_norm[ii,3],Input_vec_norm[ii,4],Input_vec_norm[ii,5],Input_vec_norm[ii,6],Input_vec_norm[ii,7])) #

        return out1
