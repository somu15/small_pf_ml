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

class MaterialModel:

    def HF(self, Ex, Ez, vxy, vxz, Gxz, ux, uy, uz):

        A = np.array([[1/Ex,-vxy/Ex,-vxz/Ez,0,0,0],[-vxy/Ex,1/Ex,-vxz/Ez,0,0,0],[-vxz/Ez,-vxz/Ez,1/Ez,0,0,0],[0,0,0,1/Gxz,0,0],[0,0,0,0,1/Gxz,0],[0,0,0,0,0,((2*(1+vxy))/Ex)]])
        A1 = np.linalg.inv(A)

        file1 = open('/Users/dhulls/projects/moose/modules/tensor_mechanics/test/tests/0_Cylinder/Cylinder_HF_R1.i', 'r')
        Lines = file1.readlines()
        Lines[97] = "    "+"value = '"+str(ux)+"'\n"
        Lines[103] = "    "+"value = '"+str(uy)+"'\n"
        Lines[109] = "    "+"value = '"+str(uz)+"'\n"
        Lines[116] = "    "+"C_ijkl = '"+str(A1[0,0])+" "+str(A1[0,1])+" "+str(A1[0,2])+" "+str(A1[2,2])+" "+str(A1[3,3])+"'\n"
        # Lines[135] = "    "+"y = '0.0 "+str(1/3*ux)+ " " +str(2/3*ux) + " " +str(ux)+"'\n"
        # Lines[140] = "    "+"y = '0.0 "+str(1/3*uy)+ " " +str(2/3*uy) + " " +str(uy)+"'\n"
        # Lines[145] = "    "+"y = '0.0 "+str(1/3*uz)+ " " +str(2/3*uz) + " " +str(uz)+"'\n"
        # Lines[152] = "    "+"C_ijkl = '"+str(A1[0,0])+" "+str(A1[0,1])+" "+str(A1[0,2])+" "+str(A1[2,2])+" "+str(A1[3,3])+"'\n"

        file1 = open('/Users/dhulls/projects/moose/modules/tensor_mechanics/test/tests/0_Cylinder/Cylinder_HF_R1.i', 'w')
        file1.writelines(Lines)
        file1.close()

        os.chdir('/Users/dhulls/projects/moose/modules/tensor_mechanics/test/tests/0_Cylinder')
        os.system('mpiexec -n 3 /Users/dhulls/projects/moose/modules/tensor_mechanics/tensor_mechanics-opt -i Cylinder_HF_R1.i')

        path1 = '/Users/dhulls/projects/moose/modules/tensor_mechanics/test/tests/0_Cylinder/Cylinder_HF_R1.csv'
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
