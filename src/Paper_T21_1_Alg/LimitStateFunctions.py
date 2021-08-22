#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 09:55:13 2020

@author: dhulls
"""

import numpy as np
from scipy.stats import norm
from TrisoModel import TrisoModel as TM

class LimitStateFunctions:

    def Triso_2d(self, Input_vec=None):

        # Input_vec = [E, v, ux]
        #             [0, 1,  2]

        siz1 = len(Input_vec[:,0])
        out1 = np.zeros(siz1)
        TM1 = TM()

        for ii in np.arange(0,siz1,1):
            out1[ii] = (TM1.Triso_2d(Input_vec[ii,0],Input_vec[ii,1],Input_vec[ii,2],Input_vec[ii,3],Input_vec[ii,4],Input_vec[ii,5]))

        return out1

    def Triso_1d(self, Input_vec=None):

        siz1 = len(Input_vec[:,0])
        out1 = np.zeros(siz1)
        TM1 = TM()

        for ii in np.arange(0,siz1,1):
            out1[ii] = (TM1.Triso_1d(Input_vec[ii,0],Input_vec[ii,1],Input_vec[ii,2],Input_vec[ii,3],Input_vec[ii,4],Input_vec[ii,5],Input_vec[ii,6])) # ,Input_vec[ii,7]

        return out1
