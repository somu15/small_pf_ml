#!/bin/bash
#PBS -M Som.Dhulipala@inl.gov
#PBS -m abe
#PBS -N TRISO_1D_T21_1
#PBS -P moose
#PBS -l select=1:ncpus=1:ngpus=2
#PBS -q gpu
#PBS -l walltime=150:00:00

JOB_NUM=${PBS_JOBID%%\.*}

cd $PBS_O_WORKDIR

\rm -f out
date > out
module load tensorflow/2.4_gpu
MV2_ENABLE_AFFINITY=0 python Alg.py >> out
date >> out
