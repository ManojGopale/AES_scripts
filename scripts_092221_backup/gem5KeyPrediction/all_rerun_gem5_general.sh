## rerun54_4p1_100 -> 4p1 rerun model51 with no batch norm for freeze layer trials
## rerun54_4p2_100 -> 4p2 rerun model51 with no batch norm for freeze layer trials

configList=(3p1 3p2 3p3 3p4 4p1 4p2 4p3 4p4 5p1 5p2 5p3 5p4 5p3_15000TrainSize 4p2_15000TrainSize 3p1_1361 3p2_1361)
oceloteRun=0
pumaRun=1
typeOfStd="col"

trainSize=7500
devSize=500
numConfig=3 ## This is not being used now
#NOTE:  need to work on the script
for configIndex in {0..11}
do
	for index in {100..100}
	do
		#Evaluates the configuration
	  #jobName=rerun_prev_config"${configList[$configIndex]}"_run_${index}_1361input_row
	  jobName=rerun_dataEns49_43_config_tr7500_"${configList[$configIndex]}"
	  echo "$jobName"
		if [[ $oceloteRun -eq 1 ]]
			then
			#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
	  	qsubCmd="qsub -N ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt -v modelName=${jobName},numPowerTraces=1500,configName=config${configList[$configIndex]} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/rerun_aes_gem5_general_template.sh"
			echo "$qsubCmd"
			eval "$qsubCmd"
		elif [[ $pumaRun -eq 1 ]]
			then
	  	sbatchCmd="sbatch --job-name ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt --export=modelName=${jobName},numPowerTraces=1500,trainSize=${trainSize},devSize=${devSize},configName=config${configList[$configIndex]},typeOfStd=${typeOfStd} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/rerun_aes_gem5_general_template_puma.sh"
			echo "$sbatchCmd"
			eval "$sbatchCmd"
		fi
		
	done
done
