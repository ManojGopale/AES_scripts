#!/bin/csh

### Set the job name 
#PBS -N chipWhisperer_bs_2048_3HL_2000_1000_1500_0p2_0p2_0p2_0p2_50patience

## Request email when job begins and ends
#PBS -m bea

########## Error and Output Path
#PBS -e /home/u3/manojgopale/AES_error_output_dir/chipWhisperer_bs_2048_3HL_2000_1000_1500_0p2_0p2_0p2_0p2_50patience_error.txt
#PBS -o /home/u3/manojgopale/AES_error_output_dir/chipWhisperer_bs_2048_3HL_2000_1000_1500_0p2_0p2_0p2_0p2_50patience_output.txt

## Specify email address for notification
#PBS -M manojgopale@email.arizona.edu

## Specify the PI group found with va command
#PBS -W group_list=rlysecky

### Set the queue to submit this job
#### We can use windfall or standard, use standard if job cannot be submitted
#PBS -q standard

### Set the number of cores and memory that will be used for this job
### select=1 is the node count, ncpus=4 are the cores in each node, 
### mem=4gb is memory per node, pcmem=6gb is the memory per core - optional
#PBS -l select=1:ncpus=28:mem=168gb:ngpus=1
###PBS -l select=1:ncpus=28:mem=168gb

### Important!!! Include this line for your small jobs.
### Without it, the entire node, containing 28 cores, will be allocated
###PBS -l place=pack:shared

## Recommended Settings
#PBS -l place=free:shared

### Specify up to maximum of 1600hours totoal cpu time for job
#PBS -l cput=1000:0:0 

### Specify upto a maximum of 240 hours walltime for job
#PBS -l walltime=70:0:0 


cd /home/u3/manojgopale
#####module load singularity
module load singularity

date
/usr/bin/time singularity exec --nv /xdisk/bethard/mig2020/extra/manojgopale/AES_data/dockerImage/ocelote_keras-2.2.4+tensorflow-1.13.1_gpu-cp35-cuda10.0-cudnn7.5.sif python3.5 /xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/scr/run_5hl_template.py --batchSize 2048 --modelName chipWhisper_bs_2048_1000epochs_100patience  --testFlag 1 --drop1 0.2 --drop2 0.2 --drop3 0.2 --drop4 0.2 --drop5 0.2
###### None of the options are required, since defaults are set in the file
########--trainSize 15000 --modelDir ${modelDir} --testDir ${testDir} --modelName ${modelName} --trainFlag ${trainFlag} --devFlag ${devFlag} --testFlag ${testFlag}
date
