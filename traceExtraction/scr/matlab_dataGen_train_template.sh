#!/bin/csh

### Set the job name 
#PBS -N matlab_config3p2

## Request email when job begins and ends
#PBS -m bea

###### Error and Output Path
#PBS -e /home/u3/manojgopale/GEM5_Analysis/mat_dataGen_train_config3p2_error.txt
#PBS -o /home/u3/manojgopale/GEM5_Analysis/mat_dataGen_train_config3p2_output.txt

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
####PBS -l select=1:ncpus=28:mem=168gb

### Important!!! Include this line for your small jobs.
### Without it, the entire node, containing 28 cores, will be allocated
###PBS -l place=pack:shared

## Recommended Settings
#PBS -l place=free:shared

### Specify up to maximum of 1600hours totoal cpu time for job
#PBS -l cput=1000:0:0 

### Specify upto a maximum of 240 hours walltime for job
#####PBS -l walltime=100:0:0 
#PBS -l walltime=70:0:0 


cd /home/u3/manojgopale
module load matlab/r2018a

########## Create matlab data from the file ##############
date
cd /extra/manojgopale/AES_data/MATLAB_work/scr/
mkdir -p /xdisk/manojgopale/AES/dataCollection/matResult/config3p2

########matlab -r 'qResult_folders = ["2019-06-27_23-05-41l2_l1associate8",  "2019-06-27_23-53-27l2_l1associate8", "2019-06-28_00-11-31l2_l1associate8"];outputPath = "/xdisk/manojgopale/AES/dataCollection/matResult/config4p2/";generateTrace(qResult_folders, outputPath);exit;'
matlab -r 'qResult_folders = ["2019-10-01_13-24-52cache_associate8"];outputPath = "/xdisk/manojgopale/AES/dataCollection/matResult/config3p2/";generateTrace(qResult_folders, outputPath);exit;'
date

########## Create csv files for traceExtraction ##############
cd /home/u3/manojgopale
module load singularity/3/3.2.1
mkdir -p /xdisk/manojgopale/AES/dataCollection/processedData/run_1_per_key/data/config3p2

date
/usr/bin/time singularity exec -B /extra/manojgopale,/xdisk/manojgopale /extra/manojgopale/AES_data/dockerImage/ocelote_keras-2.2.4+tensorflow-1.13.1_gpu-cp35-cuda10.0-cudnn7.5.sif python3.5 /extra/manojgopale/AES_data/processedData/run_1_per_key/scr/process_raw_data.py -c config3p2 -b 128 -d /xdisk/manojgopale/AES/dataCollection/matResult/config3p2/

########## Train the model and save its results ##############
mkdir -p /xdisk/manojgopale/AES/dataCollection/processedData/run_1_per_key/data/config3p2
mkdir -p /xdisk/manojgopale/AES/dataCollection/processedData/run_1_per_key/result/config3p2

/usr/bin/time singularity exec --nv /extra/manojgopale/AES_data/dockerImage/ocelote_keras-2.2.4+tensorflow-1.13.1_gpu-cp35-cuda10.0-cudnn7.5.sif python3.5 /extra/manojgopale/AES_data/traceExtraction/scr/run_1hl.py -c config3p2 --hiddenSize 1000 --modelName model1_fib_500_wk
date
##############################################################
