#!/bin/bash

### Set the job name 
#####PBS -N matlab_config3p1

## Request email when job begins and ends
#PBS -m bea

###### Error and Output Path
######PBS -e /home/u3/manojgopale/GEM5_Analysis/matlab_2020_config3p1_error.txt
######PBS -o /home/u3/manojgopale/GEM5_Analysis/matlab_2020_config3p1_output.txt

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
#######ELgato
#####PBS -l select=1:ncpus=16:mem=62gb:pcmem=4gb

### Important!!! Include this line for your small jobs.
### Without it, the entire node, containing 28 cores, will be allocated
###PBS -l place=pack:shared

## Recommended Settings
#PBS -l place=free:shared

### Specify up to maximum of 1600hours totoal cpu time for job
#PBS -l cput=500:0:0 

### Specify upto a maximum of 240 hours walltime for job
#####PBS -l walltime=100:0:0 
#PBS -l walltime=20:0:0 


cd /home/u3/manojgopale
module load matlab

date
cd /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/MATLAB_work/scr/
#####mkdir -p /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p1

########matlab -r 'qResult_folders = ["2019-06-27_23-05-41l2_l1associate8",  "2019-06-27_23-53-27l2_l1associate8", "2019-06-28_00-11-31l2_l1associate8"];outputPath = "/xdisk/manojgopale/AES/dataCollection/matResult/config4p2/";generateTrace(qResult_folders, outputPath);exit;'
export manoj_startKey=${startKey}
export manoj_endKey=${endKey}
export manoj_configName=${configName}
matlab -r "try;startKey=str2double(getenv('manoj_startKey'));endKey=str2double(getenv('manoj_endKey'));configName=string(getenv('configName'));fprintf('startKey=%d endKey= %d, configName=%s',startKey, endKey, configName);qResult_folders = [string('l24m_l1associate8_0'), 'l24m_l1associate8_1', 'l24m_l1associate8_2'];outputPath = '/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/' + configName';fprintf('\noutputPath=%s\n', outputPath);parfor key=startKey:endKey; generateTrace(qResult_folders, outputPath, key, key);end;catch;end;exit;"
date

####cd /home/u3/manojgopale
####module load singularity
####mkdir -p /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/config3p1/
####
####date
####/usr/bin/time singularity exec -B /xdisk/rlysecky/manojgopale/ /xdisk/rlysecky/manojgopale/extra/fromBethard/AES_data/dockerImage/ocelote_keras-2.2.4+tensorflow-1.13.1_gpu-cp35-cuda10.0-cudnn7.5.sif python3.5 /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/import_mat_to_python_csv_peak.py -c config3p1  -d /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/ --partLen 32
####date
