#!/bin/bash

### Set the job name 
####PBS -N bert_umls_b23_noBiobert

## Request email when job begins and ends
#PBS -m bea

###### Error and Output Path
#####PBS -e /home/u3/manojgopale/BERT_Analysis/bert_umls_b23_noBiobert_error.txt
#####PBS -o /home/u3/manojgopale/BERT_Analysis/bert_umls_b23_noBiobert_output.txt

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
####Elgato
#####PBS -l select=1:ncpus=16:mem=62gb:pcmem=4gb

### Important!!! Include this line for your small jobs.
### Without it, the entire node, containing 28 cores, will be allocated
###PBS -l place=pack:shared

## Recommended Settings
#PBS -l place=free:shared

### Specify up to maximum of 1600hours totoal cpu time for job
#PBS -l cput=10:0:0 

### Specify upto a maximum of 240 hours walltime for job
#####PBS -l walltime=100:0:0 
#PBS -l walltime=2:0:0 


cd /home/u3/manojgopale
module load singularity

date
/usr/bin/time singularity exec -B /xdisk/rlysecky/manojgopale/xdisk/ /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/dockerImage/gem5_approx_3399input_newDirstruct.sif  python2 /manojwork/work/runs.py -r ${totalRun} -c ${configName} --cpuCount ${cpuCount}
####/usr/bin/time singularity exec -B /xdisk/rlysecky/manojgopale/xdisk/ /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/dockerImage/gem5_approx_3399input_newDirstruct.sif  /bin/bash /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/runDir/full_${configName}.sh
date
