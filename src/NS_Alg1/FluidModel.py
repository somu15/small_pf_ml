#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 09:39:23 2021

@author: dhulls
"""

from __future__ import print_function
from __future__ import absolute_import
from argparse import ArgumentParser

import os
import sys
import csv
sys.path.append('.')

import numpy as np

class FluidModel:

    def Navier_Stokes(self, viscosity, density, u_Bottom, u_Top, u_Left, u_Right):

        visc = viscosity
        dens = density
        utop = u_Top
        ubot = u_Bottom
        uright = u_Right
        uleft = u_Left

        file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/NS_R1.i', 'r')
        Lines = file1.readlines()
        Lines[113] = "    "+"prop_values = '"+str(dens)+" "+str(visc)+"'\n"
        Lines[80] = "    "+"values = '"+str(utop)+" 0.0 0.0'\n"
        Lines[86] = "    "+"values = '"+str(-ubot)+" 0.0 0.0'\n"
        Lines[92] = "    "+"values = '0.0 "+str(-uleft)+" 0.0'\n"
        Lines[98] = "    "+"values = '0.0 "+str(uright)+" 0.0'\n"

        file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/NS_R1.i', 'w')
        file1.writelines(Lines)
        file1.close()

        os.chdir('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady')
        os.system('mpiexec -n 3 /Users/dhulls/projects/moose/modules/navier_stokes/navier_stokes-opt -i NS_R1.i')

        path1 = '/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/NS_R1.csv'
        with open(path1) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            Samp0 = []
            Samp1 = []
            count = 0
            for row in readCSV:
                if count > 0:
                    Samp0.append(float(row[1]))
                    Samp1.append(float(row[2]))
                count = count + 1

        resultant_NS = np.sqrt(Samp0[1]**2+Samp1[1]**2)
        return resultant_NS

    def Navier_Stokes1(self, viscosity, density, u_Bottom, u_Top, u_Left, u_Right):

        visc = viscosity
        dens = density
        utop = u_Top
        ubot = u_Bottom
        uright = u_Right
        uleft = u_Left

        file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/NS1_R1.i', 'r')
        Lines = file1.readlines()
        Lines[113] = "    "+"prop_values = '"+str(dens)+" "+str(visc)+"'\n"
        Lines[80] = "    "+"values = '"+str(utop)+" 0.0 0.0'\n"
        Lines[86] = "    "+"values = '"+str(-ubot)+" 0.0 0.0'\n"
        Lines[92] = "    "+"values = '0.0 "+str(-uleft)+" 0.0'\n"
        Lines[98] = "    "+"values = '0.0 "+str(uright)+" 0.0'\n"

        file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/NS1_R1.i', 'w')
        file1.writelines(Lines)
        file1.close()

        os.chdir('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady')
        os.system('mpiexec -n 3 /Users/dhulls/projects/moose/modules/navier_stokes/navier_stokes-opt -i NS1_R1.i')

        path1 = '/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/NS1_R1.csv'
        with open(path1) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            Samp0 = []
            Samp1 = []
            count = 0
            for row in readCSV:
                if count > 0:
                    Samp0.append(float(row[1]))
                    Samp1.append(float(row[2]))
                count = count + 1

        resultant_NS = np.sqrt(Samp0[1]**2+Samp1[1]**2)
        return resultant_NS

    def Stokes(self, viscosity, density, u_Bottom, u_Top, u_Left, u_Right):

        visc = viscosity
        dens = density
        utop = u_Top
        ubot = u_Bottom
        uright = u_Right
        uleft = u_Left

        file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/S_R1.i', 'r')
        Lines = file1.readlines()
        Lines[109] = "    "+"prop_values = '"+str(dens)+" "+str(visc)+"'\n"
        Lines[76] = "    "+"values = '"+str(utop)+" 0.0 0.0'\n"
        Lines[82] = "    "+"values = '"+str(-ubot)+" 0.0 0.0'\n"
        Lines[88] = "    "+"values = '0.0 "+str(-uleft)+" 0.0'\n"
        Lines[94] = "    "+"values = '0.0 "+str(uright)+" 0.0'\n"

        file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/S_R1.i', 'w')
        file1.writelines(Lines)
        file1.close()

        os.chdir('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady')
        os.system('/Users/dhulls/projects/moose/modules/navier_stokes/navier_stokes-opt -i S_R1.i')

        path1 = '/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/S_R1.csv'
        with open(path1) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            Samp0 = []
            Samp1 = []
            count = 0
            for row in readCSV:
                if count > 0:
                    Samp0.append(float(row[1]))
                    Samp1.append(float(row[2]))
                count = count + 1

        resultant_S = np.sqrt(Samp0[1]**2+Samp1[1]**2)
        return resultant_S

    def Stokes1(self, viscosity, density, u_Bottom, u_Top, u_Left, u_Right):

        visc = viscosity
        dens = density
        utop = u_Top
        ubot = u_Bottom
        uright = u_Right
        uleft = u_Left

        file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/S1_R1.i', 'r')
        Lines = file1.readlines()
        Lines[109] = "    "+"prop_values = '"+str(dens)+" "+str(visc)+"'\n"
        Lines[76] = "    "+"values = '"+str(utop)+" 0.0 0.0'\n"
        Lines[82] = "    "+"values = '"+str(-ubot)+" 0.0 0.0'\n"
        Lines[88] = "    "+"values = '0.0 "+str(-uleft)+" 0.0'\n"
        Lines[94] = "    "+"values = '0.0 "+str(uright)+" 0.0'\n"

        file1 = open('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/S1_R1.i', 'w')
        file1.writelines(Lines)
        file1.close()

        os.chdir('/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady')
        os.system('/Users/dhulls/projects/moose/modules/navier_stokes/navier_stokes-opt -i S1_R1.i')

        path1 = '/Users/dhulls/projects/moose/modules/navier_stokes/test/tests/ins/2d_steady/S1_R1.csv'
        with open(path1) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            Samp0 = []
            Samp1 = []
            count = 0
            for row in readCSV:
                if count > 0:
                    Samp0.append(float(row[1]))
                    Samp1.append(float(row[2]))
                count = count + 1

        resultant_S = np.sqrt(Samp0[1]**2+Samp1[1]**2)
        return resultant_S
