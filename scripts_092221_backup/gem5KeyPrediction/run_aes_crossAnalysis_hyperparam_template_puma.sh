#!/bin/bash

### Set the job name 
#####SBATCH --job-name matlab_config3p1

## Request email when job begins and ends
#SBATCH --mail-user=manojgopale@email.arizona.edu
#SBATCH --mail-type=ALL

###### Error and Output Path
######SBATCH -e /home/u3/manojgopale/GEM5_Analysis/matlab_2020_config3p1_error.txt
######SBATCH -o /home/u3/manojgopale/GEM5_Analysis/matlab_2020_config3p1_output.txt

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
##Puma : Singularity is already loaded
#####module load singularity

date
######Used for hyper-tuning, dev set utilized
#####/usr/bin/time singularity exec --nv /xdisk/rlysecky/manojgopale/extra/fromBethard/AES_data/dockerImage/ocelote_keras-2.2.4+tensorflow-1.13.1_gpu-cp35-cuda10.0-cudnn7.5.sif  python3.5 /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/crossAnalysis_hyperparam_tuning.py --devDataConfig ${devDataConfig} --trainSize ${trainSize} --devSize ${devSize} --modelDir ${modelDir} --devDataDir ${devDataDir} --trainFlag ${trainFlag} --devFlag ${devFlag} --testFlag ${testFlag} --modelConfig ${modelConfig} --inputTraces ${inputTraces} --typeOfStd ${typeOfStd}

######USed to test final models with test set
/usr/bin/time singularity exec --nv /xdisk/rlysecky/manojgopale/extra/fromBethard/AES_data/dockerImage/ocelote_keras-2.2.4+tensorflow-1.13.1_gpu-cp35-cuda10.0-cudnn7.5.sif  python3.5 /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/crossAnalysis_hyperparam_tuning.py --testDataConfig ${testDataConfig} --trainSize ${trainSize} --testSize ${testSize} --modelDir ${modelDir} --testDataDir ${testDataDir} --trainFlag ${trainFlag} --devFlag ${devFlag} --testFlag ${testFlag} --modelConfig ${modelConfig} --inputTraces ${inputTraces} --typeOfStd ${typeOfStd}
date
