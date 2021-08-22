#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 15:34:11 2020

@author: dhulls
"""

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
from pyDOE import *

class DrawRandom:

    def TrisoRandom(self,N=1):
        out = np.zeros((N,7))
        rv1 = norm(loc=213.35e-6,scale=4.4e-6) # kernel_r
        rv2 = norm(loc=98.9e-6,scale=8.4e-6) # buffer_t
        rv3 = norm(loc=40.4e-6,scale=2.5e-6) # ipyc_t
        rv4 = norm(loc=35.2e-6,scale=1.2e-6) # sic_t
        rv5 = norm(loc=43.4e-6,scale=2.9e-6) # opyc_t
        rv6 = uniform()
        out[:,0] = rv1.rvs(size=N)
        out[:,1] = (rv2.rvs(size=N))
        out[:,2] = (rv3.rvs(size=N))
        out[:,3] = (rv4.rvs(size=N))
        out[:,4] = (rv5.rvs(size=N))
        out[:,5] = (rv6.rvs(size=N))
        out[:,6] = (rv6.rvs(size=N))
        # out[:,7] = (rv6.rvs(size=N))
        return out

    def TrisoLHS(self, Nsamps=None):
        out = np.zeros((Nsamps,7))
        lhd0 = lhs(7, samples=Nsamps, criterion='centermaximin')
        out[:,0] = uniform(loc=0.0001938302254231716,scale=3.5111867415040426e-05).ppf(lhd0[:,0]) # kernel_r
        out[:,1] = uniform(loc=7.27370536376046e-05,scale=5.230937638444862e-05).ppf(lhd0[:,1]) # buffer_t
        out[:,2] = uniform(loc=3.270382725868163e-05,scale=1.5642320384140967e-05).ppf(lhd0[:,2]) # ipyc_t
        out[:,3] = uniform(loc=3.072962740713149e-05,scale=7.42526553558767e-06).ppf(lhd0[:,3]) # sic_t
        out[:,4] = uniform(loc=3.5473696792826675e-05,scale=2.0917204816879883e-05).ppf(lhd0[:,4]) # opyc_t
        out[:,5] = uniform().ppf(lhd0[:,5])
        out[:,6] = uniform().ppf(lhd0[:,6])
        # out[:,7] = uniform().ppf(lhd0[:,7])
        # out1 = np.zeros((Nsamps,6))
        # out1[:,0] = out[:,0]
        # out1[:,1] = out[:,1]
        # out1[:,2] = out[:,2]
        # out1[:,3] = out[:,3]
        # out1[:,4] = out[:,4]
        # out1[:,5] = out[:,7]
        return out

    def TrisoPDF(self, rv_req=None, index=None):

        if index == 0:
            rv = norm(loc=213.35e-6,scale=4.4e-6) # kernel_r
            out = rv.pdf((rv_req))
        elif index == 1:
            rv = norm(loc=98.9e-6,scale=8.4e-6) # buffer_t
            out = rv.pdf((rv_req))
        elif index == 2:
            rv = norm(loc=40.4e-6,scale=2.5e-6) # ipyc_t
            out = rv.pdf((rv_req))
        elif index == 3:
            rv = norm(loc=35.2e-6,scale=1.2e-6) # sic_t
            out = rv.pdf((rv_req))
        elif index == 4:
            rv = norm(loc=43.4e-6,scale=2.9e-6) # opyc_t
            out = rv.pdf((rv_req))
        else:
            rv = uniform()
            out = rv.pdf((rv_req))

        return out
