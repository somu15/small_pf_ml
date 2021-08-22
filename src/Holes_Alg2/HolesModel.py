#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 12:37:06 2021

@author: dhulls
"""

from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import csv
sys.path.append('.')

class HolesModel:

    def HF(self, E1, v1, ux):

        file1 = open('/Users/dhulls/projects/moose/modules/tensor_mechanics/test/tests/0_Holes/HF_R2.i', 'r')
        Lines = file1.readlines()
        Lines[101] = "    "+"value = '"+str(ux)+"'\n"
        Lines[114] = "    "+"youngs_modulus = "+str(E1)+"\n"
        Lines[115] = "    "+"poissons_ratio = "+str(v1)+"\n"

        file1 = open('/Users/dhulls/projects/moose/modules/tensor_mechanics/test/tests/0_Holes/HF_R2.i', 'w')
        file1.writelines(Lines)
        file1.close()

        os.chdir('/Users/dhulls/projects/moose/modules/tensor_mechanics/test/tests/0_Holes')
        os.system('mpiexec -n 3 /Users/dhulls/projects/moose/modules/tensor_mechanics/tensor_mechanics-opt -i HF_R2.i')

        path1 = '/Users/dhulls/projects/moose/modules/tensor_mechanics/test/tests/0_Holes/HF_R2.csv'
        with open(path1) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            Samp0 = []
            count = 0
            for row in readCSV:
                if count > 0:
                    Samp0.append(float(row[1]))
                count = count + 1

        stress_von = Samp0[1]

        return stress_von

    def HF1(self, E1, v1, ux):

        file1 = open('/Users/dhulls/projects/moose/modules/tensor_mechanics/test/tests/0_Holes/HF1_R2.i', 'r')
        Lines = file1.readlines()
        Lines[101] = "    "+"value = '"+str(ux)+"'\n"
        Lines[114] = "    "+"youngs_modulus = "+str(E1)+"\n"
        Lines[115] = "    "+"poissons_ratio = "+str(v1)+"\n"

        file1 = open('/Users/dhulls/projects/moose/modules/tensor_mechanics/test/tests/0_Holes/HF1_R2.i', 'w')
        file1.writelines(Lines)
        file1.close()

        os.chdir('/Users/dhulls/projects/moose/modules/tensor_mechanics/test/tests/0_Holes')
        os.system('mpiexec -n 3 /Users/dhulls/projects/moose/modules/tensor_mechanics/tensor_mechanics-opt -i HF1_R2.i')

        path1 = '/Users/dhulls/projects/moose/modules/tensor_mechanics/test/tests/0_Holes/HF1_R2.csv'
        with open(path1) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            Samp0 = []
            count = 0
            for row in readCSV:
                if count > 0:
                    Samp0.append(float(row[1]))
                count = count + 1

        stress_von = Samp0[1]

        return stress_von
