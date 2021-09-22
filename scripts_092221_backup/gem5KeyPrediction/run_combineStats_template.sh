#!/bin/bash

### Set the job name 
#PBS -N combineStats_all

## Request email when job begins and ends
#PBS -m bea

########## Error and Output Path
#PBS -e /home/u3/manojgopale/GEM5_Analysis/combineStats_all_error.txt
#PBS -o /home/u3/manojgopale/GEM5_Analysis/combineStats_all_output.txt

## Specify email address for notification
#PBS -M manojgopale@email.arizona.edu

## Specify the PI group found with va command
#PBS -W group_list=rlysecky

### Set the queue to submit this job
#### We can use windfall or standard, use standard if job cannot be submitted
#PBS -q windfall

### Set the number of cores and memory that will be used for this job
### select=1 is the node count, ncpus=4 are the cores in each node, 
### mem=4gb is memory per node, pcmem=6gb is the memory per core - optional
#PBS -l select=1:ncpus=28:mem=168gb:ngpus=1
####PBS -l select=1:ncpus=2:mem=336gb:pcmem=42gb
###PBS -l select=1:ncpus=28:mem=168gb

### Important!!! Include this line for your small jobs.
### Without it, the entire node, containing 28 cores, will be allocated
###PBS -l place=pack:shared

## Recommended Settings
#PBS -l place=free:shared

### Specify up to maximum of 1600hours totoal cpu time for job
#PBS -l cput=100:0:0 

### Specify upto a maximum of 240 hours walltime for job
#PBS -l walltime=10:0:0 


cd /home/u3/manojgopale
#####module load singularity
module load singularity

date
/usr/bin/time singularity exec --nv /xdisk/rlysecky/manojgopale/extra/fromBethard/AES_data/dockerImage/ocelote_keras-2.2.4+tensorflow-1.13.1_gpu-cp35-cuda10.0-cudnn7.5.sif python3.5 /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/combineStats.py
date
