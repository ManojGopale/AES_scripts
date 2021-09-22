#!/bin/bash

### Set the job name 
#SBATCH --job-name S3_traceDiff

## Request email when job begins and ends
#SBATCH --mail-user=manojgopale@email.arizona.edu
#SBATCH --mail-type=ALL

###### Error and Output Path
#SBATCH -e /home/u3/manojgopale/GEM5_Analysis/S3_traceDiff_error.txt
#SBATCH -o /home/u3/manojgopale/GEM5_Analysis/S3_traceDiff_output.txt

## Specify the PI group found with va command
#SBATCH --account=rlysecky

### Set the queue to submit this job
#### We can use windfall or standard, use standard if job cannot be submitted
#SBATCH --partition=standard

### Set the number of cores and memory that will be used for this job
#SBATCH --nodes=1
#SBATCH --ntasks=25
#SBATCH --mem=100gb

#SBATCH --exclusive

### Specify upto a maximum of 240 hours walltime for job
#SBATCH --time 5:0:0

cd /home/u3/manojgopale
#####module load singularity
####module load singularity

date
#######numConfigs should correspond to the configs in baselineList
/usr/bin/time singularity exec --nv /xdisk/rlysecky/manojgopale/extra/fromBethard/AES_data/dockerImage/ocelote_keras-2.2.4+tensorflow-1.13.1_gpu-cp35-cuda10.0-cudnn7.5.sif python3.5 /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/trace_differences.py --baselineList config3p2,config4p2 --targetList config3p1 --numConfigs 2
date
