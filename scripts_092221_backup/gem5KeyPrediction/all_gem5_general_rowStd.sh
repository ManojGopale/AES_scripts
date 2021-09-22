## config3p1
## runs 0-10 -> hidden layers power of 2, adam default settings
## runs 6-10 -> 32-515 nodes/HL, 1-4 HL, reducing complexity of runs
## runs 11-15 -> with 10000 per trace
## runs 16-20 -> with 5000 per trace
## runs 21-25 -> with 3000 per trace
configList=(3p1 3p2 3p3 3p4 4p1 4p2 4p3 4p4 5p1 5p2 5p3 5p4 5p3_15000TrainSize 3p1_1361 3p2_1361)
for configIndex in {0..0}
do
	for index in {21..25}
	do
		#Evaluates the configuration
	  jobName=run_rowStd_config"${configList[${configIndex}]}"_"${index}"
	  echo "$jobName"
		#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
	  qsubCmd="qsub -N ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt -v modelName=${jobName},numPowerTraces=1500,configName=config${configList[$configIndex]} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/run_aes_gem5_rowStd_template.sh"
		echo "$qsubCmd"
		eval "$qsubCmd"
	done
done
