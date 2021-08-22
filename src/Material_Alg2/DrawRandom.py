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

    # def __init__(self):
    #     # self.Input_vec = Input_vec

    def StandardNormal_Indep(self, N=None):
        out = np.zeros(N)
        rv = norm(loc=0,scale=1)
        for ii in np.arange(0,N,1):
            out[ii] = rv.rvs()
        return out

    def BoreholeRandom(self):
        out = np.zeros(8)
        rv1 = norm(loc=(7.71),scale=1.0056)
        rv2 = uniform()
        out[0] = 0.05 + 0.1 * rv2.rvs()
        out[1] = (rv1.rvs())
        out[2] = 63070 + 52530 * rv2.rvs()
        out[3] = 990 + 120 * rv2.rvs()
        out[4] = 63.1 + 52.9 * rv2.rvs()
        out[5] = 700 + 120 * rv2.rvs()
        out[6] = 1120 + 560 * rv2.rvs()
        out[7] = 9855 + 2190 * rv2.rvs()
        return out


    def BoreholePDF(self, rv_req=None, index=None):
        if index == 0:
            rv = uniform(loc=0.05,scale=0.1)
            out = rv.pdf((rv_req))
        elif index == 1:
            rv = norm(loc=(7.71),scale=1.0056)
            out = rv.pdf((rv_req))
        elif index == 2:
            rv = uniform(loc=63070,scale=52530)
            out = rv.pdf((rv_req))
        elif index == 3:
            rv = uniform(loc = 990, scale=120)
            out = rv.pdf((rv_req))
        elif index == 4:
            rv = uniform(loc=63.1,scale=52.9)
            out = rv.pdf((rv_req))
        elif index == 5:
            rv = uniform(loc=700,scale=120)
            out = rv.pdf((rv_req))
        elif index == 6:
            rv = uniform(loc=1120,scale=560)
            out = rv.pdf((rv_req))
        else:
            rv = uniform(loc=9855,scale=2190)
            out = rv.pdf((rv_req))
        return out

    def BoreholeLHS(self, Nsamps=None):
        out = np.zeros((Nsamps,8))
        lhd0 = lhs(8, samples=Nsamps, criterion='centermaximin')
        out[:,0] = uniform(loc=0.05,scale=0.1).ppf(lhd0[:,0])
        out[:,1] = uniform(loc=4.0,scale=11.0).ppf(lhd0[:,1])  # norm(loc=np.log(7.71),scale=1.0056).ppf(lhd0[:,1])
        out[:,2] = uniform(loc=63070,scale=52530).ppf(lhd0[:,2])
        out[:,3] = uniform(loc = 990, scale=120).ppf(lhd0[:,3])
        out[:,4] = uniform(loc=63.1,scale=52.9).ppf(lhd0[:,4])
        out[:,5] = uniform(loc=700,scale=120).ppf(lhd0[:,5])
        out[:,6] = uniform(loc=1120,scale=560).ppf(lhd0[:,6])
        out[:,7] = uniform(loc=9855,scale=2190).ppf(lhd0[:,7])
        return out

    def TrussRandom(self):
        # out = np.zeros(10)
        # rv1 = norm(loc=26.0653,scale=0.099) # E1 and E2
        # rv2 = norm(loc=-6.2195,scale=0.09975) # A1
        # rv3 = norm(loc=-6.9127,scale=0.099503) # A2
        # rv4 = gumbel_r(loc=5e4,scale=(7.5e3/1.28254)) # P1, P2, P3, P4, P5, and P6
        # out[0] = np.exp(rv1.rvs())
        # out[1] = np.exp(rv1.rvs())
        # out[2] = np.exp(rv2.rvs())
        # out[3] = np.exp(rv3.rvs())
        # out[4] = rv4.rvs()
        # out[5] = rv4.rvs()
        # out[6] = rv4.rvs()
        # out[7] = rv4.rvs()
        # out[8] = rv4.rvs()
        # out[9] = rv4.rvs()
        out = np.zeros(10)
        rv1 = norm(loc=26.0653,scale=0.099) # E1 and E2
        rv2 = norm(loc=-6.2195,scale=0.09975) # A1
        rv3 = norm(loc=-6.9127,scale=0.099503) # A2
        # rv4 = gumbel_r(loc=5e4,scale=(7.5e3/1.28254)) # P1, P2, P3, P4, P5, and P6
        rv4 = gumbel_l(loc=46625.862,scale=(5847.726)) # P1, P2, P3, P4, P5, and P6
        out[0] = np.exp(rv1.rvs())
        out[1] = np.exp(rv1.rvs())
        out[2] = np.exp(rv2.rvs())
        out[3] = np.exp(rv3.rvs())
        out[4] = rv4.rvs()
        out[5] = rv4.rvs()
        out[6] = rv4.rvs()
        out[7] = rv4.rvs()
        out[8] = rv4.rvs()
        out[9] = rv4.rvs()
        return out

    def TrussPDF(self, rv_req=None, index=None):
        if index == 0 or index==1:
            rv = norm(loc=26.0653,scale=0.099) # E1 and E2
            out = rv.pdf(np.log(rv_req))
        elif index == 2:
            rv = norm(loc=-6.2195,scale=0.09975) # A1
            out = rv.pdf(np.log(rv_req))
        elif index==3:
            rv = norm(loc=-6.9127,scale=0.099503) # A2
            out = rv.pdf(np.log(rv_req))
        else:
            rv = gumbel_l(loc=46625.862,scale=(5847.726)) # gumbel_l(loc=5e4,scale=(7.5e3/1.28254)) # P1, P2, P3, P4, P5, and P6
            out = rv.pdf((rv_req))
        return out

    def TrussLHS(self, Nsamps=None):
        out = np.zeros((Nsamps,10))
        lhd0 = lhs(10, samples=Nsamps, criterion='centermaximin')
        out[:,0] = uniform(loc=1.365e11,scale=1.47e11).ppf(lhd0[:,0]) # uniform(loc=25.7188,scale=0.693).ppf(lhd0[:,0])
        out[:,1] = uniform(loc=1.365e11,scale=1.47e11).ppf(lhd0[:,1]) # uniform(loc=25.7188,scale=0.693).ppf(lhd0[:,1])
        out[:,2] = uniform(loc=1.3e-3,scale=1.4e-3).ppf(lhd0[:,2]) # uniform(loc=-6.56862,scale=0.69824).ppf(lhd0[:,2])
        out[:,3] = uniform(loc=6.5e-4,scale=7e-4).ppf(lhd0[:,3]) # uniform(loc=-7.26095, scale=0.6965).ppf(lhd0[:,3])
        out[:,4] = uniform(loc=23750,scale=52500).ppf(lhd0[:,4]) # uniform(loc=23750,scale=52500).ppf(lhd0[:,4])
        out[:,5] = uniform(loc=23750,scale=52500).ppf(lhd0[:,5])
        out[:,6] = uniform(loc=23750,scale=52500).ppf(lhd0[:,6])
        out[:,7] = uniform(loc=23750,scale=52500).ppf(lhd0[:,7])
        out[:,8] = uniform(loc=23750,scale=52500).ppf(lhd0[:,8])
        out[:,9] = uniform(loc=23750,scale=52500).ppf(lhd0[:,9])
        return out

    def MaterialRandom(self,N=1):
        out = np.zeros((N,8))
        rv1 = norm(loc=np.log(200),scale=0.1) # Ex
        rv2 = norm(loc=np.log(300),scale=0.1) # Ez
        rv3 = norm(loc=np.log(0.25),scale=0.1) # vxy
        rv4 = norm(loc=np.log(0.3),scale=0.1) # vxz
        rv5 = norm(loc=np.log(135),scale=0.1) # Gxz
        rv6 = norm(loc=np.log(0.15),scale=0.5) # ux, uy, uz
        out[:,0] = np.exp(rv1.rvs(size=N))
        out[:,1] = np.exp(rv2.rvs(size=N))
        out[:,2] = np.exp(rv3.rvs(size=N))
        out[:,3] = np.exp(rv4.rvs(size=N))
        out[:,4] = np.exp(rv5.rvs(size=N))
        out[:,5] = np.exp(rv6.rvs(size=N))
        out[:,6] = np.exp(rv6.rvs(size=N))
        out[:,7] = np.exp(rv6.rvs(size=N))
        out1 = np.zeros((N,5))
        out1[:,0] = out[:,0]
        out1[:,1] = out[:,2]
        out1[:,2] = out[:,5]
        out1[:,3] = out[:,6]
        out1[:,4] = out[:,7]
        return out, out1

    def MaterialLHS(self, Nsamps=None):
        out = np.zeros((Nsamps,8))
        lhd0 = lhs(8, samples=Nsamps, criterion='centermaximin')
        out[:,0] = uniform(loc=140.937,scale=142.876).ppf(lhd0[:,0]) # Ex
        out[:,1] = uniform(loc=211.406,scale=214.314).ppf(lhd0[:,1]) # Ez
        out[:,2] = uniform(loc=0.176,scale=0.178).ppf(lhd0[:,2]) # vxy
        out[:,3] = uniform(loc=0.211,scale=0.214).ppf(lhd0[:,3]) # vxz
        out[:,4] = uniform(loc=95.132,scale=96.442).ppf(lhd0[:,4]) # Gxz
        out[:,5] = uniform(loc=0.05,scale=0.75).ppf(lhd0[:,5]) # ux
        out[:,6] = uniform(loc=0.05,scale=0.75).ppf(lhd0[:,6]) # uy
        out[:,7] = uniform(loc=0.05,scale=0.75).ppf(lhd0[:,7]) # uz
        out1 = np.zeros((Nsamps,5))
        out1[:,0] = out[:,0]
        out1[:,1] = out[:,2]
        out1[:,2] = out[:,5]
        out1[:,3] = out[:,6]
        out1[:,4] = out[:,7]
        return out, out1

    def MaterialPDF(self, rv_req=None, index=None, LF=None):

        if LF==0:

            if index == 0:
                rv = norm(loc=np.log(200),scale=0.1) # Ex
                out = rv.pdf(np.log(rv_req))
            elif index == 1:
                rv = norm(loc=np.log(300),scale=0.1) # Ez
                out = rv.pdf(np.log(rv_req))
            elif index == 2:
                rv = norm(loc=np.log(0.25),scale=0.1) # vxy
                out = rv.pdf(np.log(rv_req))
            elif index == 3:
                rv = norm(loc=np.log(0.3),scale=0.1) # vxz
                out = rv.pdf(np.log(rv_req))
            elif index == 4:
                rv = norm(loc=np.log(135),scale=0.1) # Gxz
                out = rv.pdf(np.log(rv_req))
            else:
                rv = norm(loc=np.log(0.15),scale=0.5) # ux, uy, uz
                out = rv.pdf(np.log(rv_req))

        else:

            if index == 0:
                rv = norm(loc=np.log(200),scale=0.1) # Ex
                out = rv.pdf(np.log(rv_req))
            elif index == 1:
                rv = norm(loc=np.log(0.25),scale=0.1) # vxy
                out = rv.pdf(np.log(rv_req))
            else:
                rv = norm(loc=np.log(0.15),scale=0.5) # ux, uy, uz
                out = rv.pdf(np.log(rv_req))

        return out

    def FluidRandom(self,N=1):
        out = np.zeros((N,6))
        lower, upper = np.log(0.005), np.log(0.05)
        mu, sigma = np.log(0.025), 0.5
        rv0 = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma) # Viscosity
        rv1 = uniform(0.5,1) # Density
        lower, upper = np.log(0.5), np.log(1.5)
        mu, sigma = np.log(0.75), 0.25
        rv2 = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma) # Velocity
        out[:,0] = np.exp(rv0.rvs())
        out[:,1] = (rv1.rvs())
        out[:,2] = -np.exp(rv2.rvs()) # u_Bottom
        out[:,3] = np.exp(rv2.rvs()) # u_Top
        out[:,4] = -np.exp(rv2.rvs()) # u_Left
        out[:,5] = np.exp(rv2.rvs()) # u_Right
        return out

    def FluidLHS(self, Nsamps=None):
        out = np.zeros((Nsamps,6))
        lhd0 = lhs(6, samples=Nsamps, criterion='centermaximin')
        out[:,0] = np.exp(uniform(loc=-5.29831,scale=2.30258).ppf(lhd0[:,0])) # Viscosity
        out[:,1] = (uniform(loc=0.5,scale=1.0).ppf(lhd0[:,1])) # Density
        out[:,2] = -np.exp(uniform(loc=-0.69314,scale=1.09861).ppf(lhd0[:,2])) # u_Bottom
        out[:,3] = np.exp(uniform(loc=-0.69314,scale=1.09861).ppf(lhd0[:,3])) # u_Top
        out[:,4] = -np.exp(uniform(loc=-0.69314,scale=1.09861).ppf(lhd0[:,4])) # u_Left
        out[:,5] = np.exp(uniform(loc=-0.69314,scale=1.09861).ppf(lhd0[:,5])) # u_Right
        return out

    def FluidPDF(self, rv_req=None, index=None):

        if index == 0:
            lower, upper = np.log(0.005), np.log(0.05)
            mu, sigma = np.log(0.025), 0.5
            rv = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma) # Viscosity
            out = rv.pdf(np.log(rv_req))
        elif index == 1:
            rv = uniform(0.5,1) # Density
            out = rv.pdf((rv_req))
        else:
            lower, upper = np.log(0.5), np.log(1.5)
            mu, sigma = np.log(0.75), 0.25
            rv = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma) # Velocity
            out = rv.pdf(np.log(np.abs(rv_req)))

        return out

    def FluidCDF(self, rv_req=None, index=None):

        if index == 0:
            lower, upper = np.log(0.005), np.log(0.05)
            mu, sigma = np.log(0.025), 0.5
            rv = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma) # Viscosity
            out = rv.cdf(np.log(rv_req))
        elif index == 1:
            rv = uniform(0.5,1) # Density
            out = rv.cdf((rv_req))
        else:
            lower, upper = np.log(0.5), np.log(1.5)
            mu, sigma = np.log(0.75), 0.25
            rv = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma) # Velocity
            out = rv.cdf(np.log(np.abs(rv_req)))

        return out

    def FluidPPF(self, rv_rand=None, index=None):

        if index == 0:
            lower, upper = np.log(0.005), np.log(0.05)
            mu, sigma = np.log(0.025), 0.5
            rv = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma) # Viscosity
            out = np.exp(rv.ppf(rv_rand))
        elif index == 1:
            rv = uniform(0.5,1) # Density
            out = rv.ppf(rv_rand)
        elif index == 2 or index == 4:
            lower, upper = np.log(0.5), np.log(1.5)
            mu, sigma = np.log(0.75), 0.25
            rv = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma) # Velocity
            out = -np.exp(rv.ppf(rv_rand))
        else:
            lower, upper = np.log(0.5), np.log(1.5)
            mu, sigma = np.log(0.75), 0.25
            rv = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma) # Velocity
            out = np.exp(rv.ppf(rv_rand))

        return out

    def HolesRandom(self):

        out = np.zeros(3)

        lower, upper = np.log(100), np.log(300)
        mu, sigma = np.log(200), 0.5
        rv0 = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma) # Young's modulus
        lower, upper = np.log(0.15), np.log(0.35)
        mu, sigma = np.log(0.25), 0.5
        rv1 = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma) # Poisson ratio
        lower, upper = np.log(0.001), np.log(0.15)
        mu, sigma = np.log(0.08), 0.5
        rv2 = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma) # Displacement
        out[0] = np.exp(rv0.rvs()) # Young's modulus
        out[1] = np.exp(rv1.rvs()) # Poisson ratio
        out[2] = np.exp(rv2.rvs()) # Displacement

        return out

    def HolesPDF(self, rv_req=None, index=None):

        if index == 0:
            lower, upper = np.log(100), np.log(300)
            mu, sigma = np.log(200), 0.5
            rv = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma) # Young's modulus
            out = rv.pdf(np.log(rv_req))
        elif index == 1:
            lower, upper = np.log(0.15), np.log(0.35)
            mu, sigma = np.log(0.25), 0.5
            rv = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma) # Poisson ratio
            out = rv.pdf(np.log(rv_req))
        else:
            lower, upper = np.log(0.001), np.log(0.15)
            mu, sigma = np.log(0.08), 0.5
            rv = truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma) # Displacement
            out = rv.pdf(np.log(rv_req))

        return out

    def HolesLHS(self, Nsamps=None):

        out = np.zeros((Nsamps,3))
        lhd0 = lhs(3, samples=Nsamps, criterion='centermaximin')
        out[:,0] = np.exp(uniform(loc=4.6051,scale=1.0986).ppf(lhd0[:,0])) # Young's modulus
        out[:,1] = np.exp(uniform(loc=-1.897,scale=0.8472).ppf(lhd0[:,1])) # Poisson ratio
        out[:,2] = np.exp(uniform(loc=-6.9077,scale=5.0106).ppf(lhd0[:,2])) # Displacement

        return out
