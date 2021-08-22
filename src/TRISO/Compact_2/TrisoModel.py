#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 15:04:32 2020

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
from scipy.stats import uniform

class TrisoModel:

    # def Triso_2d(self, kr, bt, it, st, ot, q3):
    #
    #     # /Users/dhulls/projects/
    #
    #     file1 = open('/Users/dhulls/projects/bison/examples/TRISO/Paper_T21_1_Alg/2d_MC.i', 'r')
    #     Lines = file1.readlines()
    #     Lines[7] = "    "+"mean = '"+str(kr)+"'\n"
    #     Lines[14] = "    "+"mean = '"+str(bt)+"'\n"
    #     Lines[21] = "    "+"mean = '"+str(it)+"'\n"
    #     Lines[28] = "    "+"mean = '"+str(st)+"'\n"
    #     Lines[35] = "    "+"mean = '"+str(ot)+"'\n"
    #     Lines[42] = "    "+"mean = '"+str(q3)+"'\n"
    #
    #     file1 = open('/Users/dhulls/projects/bison/examples/TRISO/Paper_T21_1_Alg/2d_MC.i', 'w')
    #     file1.writelines(Lines)
    #     file1.close()
    #
    #     os.chdir('/Users/dhulls/projects/bison/examples/TRISO/Paper_T21_1_Alg')
    #     os.system('/Users/dhulls/projects/bison/bison-opt -i 2d_MC.i') # mpiexec -n 3
    #
    #     path1 = '/Users/dhulls/projects/bison/examples/TRISO/Paper_T21_1_Alg/2d_MC_out_SiC_stress_diff_0001.csv'
    #     with open(path1) as csvfile:
    #         readCSV = csv.reader(csvfile, delimiter=',')
    #         Samp0 = []
    #         count = 0
    #         for row in readCSV:
    #             if count > 0:
    #                 Samp0.append(float(row[0]))
    #             count = count + 1
    #
    #     stress_von = Samp0[0]
    #
    #     return stress_von

    def Triso_1d(self, kr, bt, it, st, ot, q1, q2, q3): #

        file1 = open('/Users/dhulls/projects/bison/examples/TRISO/Paper/Alg/Compact_2/1d_MC.i', 'r')
        Lines = file1.readlines()
        Lines[7] = "    "+"mean = '"+str(kr)+"'\n"
        Lines[14] = "    "+"mean = '"+str(bt)+"'\n"
        Lines[21] = "    "+"mean = '"+str(it)+"'\n"
        Lines[28] = "    "+"mean = '"+str(st)+"'\n"
        Lines[35] = "    "+"mean = '"+str(ot)+"'\n"
        Lines[42] = "    "+"mean = '"+str(q1)+"'\n"
        Lines[48] = "    "+"mean = '"+str(q2)+"'\n"
        # rv1 = uniform(loc=0.5,scale=0.5)
        # Lines[54] = "    "+"mean = '"+str(rv1.rvs())+"'\n"
        Lines[54] = "    "+"mean = '"+str(q3)+"'\n"

        file1 = open('/Users/dhulls/projects/bison/examples/TRISO/Paper/Alg/Compact_2/1d_MC.i', 'w')
        file1.writelines(Lines)
        file1.close()

        os.chdir('/Users/dhulls/projects/bison/examples/TRISO/Paper/Alg/Compact_2/')
        os.system('/Users/dhulls/projects/bison/bison-opt -i 1d_MC.i') # mpiexec -n 3

        path1 = '/Users/dhulls/projects/bison/examples/TRISO/Paper/Alg/Compact_2/1d_MC_out_sic_stress_return_0001.csv'
        with open(path1) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            Samp0 = []
            count = 0
            for row in readCSV:
                if count > 0:
                    Samp0.append(float(row[0]))
                count = count + 1

        stress_von = Samp0[0]

        return stress_von
