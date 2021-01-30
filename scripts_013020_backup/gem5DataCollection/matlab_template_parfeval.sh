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
#SBATCH --ntasks=94
#SBATCH --mem=470gb

#SBATCH --exclusive

### Specify upto a maximum of 240 hours walltime for job
#SBATCH --time 20:0:0

cd /home/u3/manojgopale
module load matlab

date
cd /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/MATLAB_work/scr/
#####mkdir -p /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p1

########matlab -r 'qResult_folders = ["2019-06-27_23-05-41l2_l1associate8",  "2019-06-27_23-53-27l2_l1associate8", "2019-06-28_00-11-31l2_l1associate8"];outputPath = "/xdisk/manojgopale/AES/dataCollection/matResult/config4p2/";generateTrace(qResult_folders, outputPath);exit;'
export manoj_startKey=${startKey}
export manoj_endKey=${endKey}
export manoj_configName=${configName}
matlab -r "try;startKey=str2double(getenv('manoj_startKey'));endKey=str2double(getenv('manoj_endKey'));configName=string(getenv('configName'));fprintf('startKey=%d endKey= %d, configName=%s',startKey, endKey, configName);qResult_folders = [string('l24m_l1associate8_0'), 'l24m_l1associate8_1', 'l24m_l1associate8_2'];outputPath = '/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/' + configName';fprintf('\noutputPath=%s\n', outputPath);parfor key=startKey:endKey;generateTrace(qResult_folders, outputPath, key, key) ; end; catch;end;exit;"
date

####cd /home/u3/manojgopale
####module load singularity
####mkdir -p /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/config3p1/
####
####date
####/usr/bin/time singularity exec -B /xdisk/rlysecky/manojgopale/ /xdisk/rlysecky/manojgopale/extra/fromBethard/AES_data/dockerImage/ocelote_keras-2.2.4+tensorflow-1.13.1_gpu-cp35-cuda10.0-cudnn7.5.sif python3.5 /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/import_mat_to_python_csv_peak.py -c config3p1  -d /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/ --partLen 32
####date
