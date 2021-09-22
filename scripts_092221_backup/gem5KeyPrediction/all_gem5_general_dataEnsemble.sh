##configList=(3p1 3p2 3p3 3p4 4p1 4p2 4p3 4p4 5p1 5p2 5p3 5p4 5p3_15000TrainSize)
## Make sure only one of the flags are 1, ocelote or puma
## runs 0-25 -> ran with 14000 traces/key/config. 
## runs 30-31 -> 3p1, 3p2, ran with 1000 traces/key/config, lower complexity of NN, 1-4 HL, 32-512 nodes/HL
## runs 32-35 -> allConfigs, 100 traces/key/config, lower complexity
## runs 36-37 -> 3p1, 3p4 ensemble, allPool Standard, 1000 traces/key/config, lower complexity
## runs 38-39 -> 4p1, 4p4 ensemble as runs36-37
## runs 40-41 -> 5p1, 5p4 ensemble as runs36-37
## runs 42-45 -> 3p4, 4p4, 5p4, other settings as runs 36-37
## runs 46-50 -> 3p3, 3p4, 4p3, 4p4, 5p3, 5p4 ensemble, with higher configs of the sets in each
## runs 51-55 -> p2 and p3 from each of the configs
## runs 56-60 -> 3p3, 4p4, 5p1, 5p3
## runs 61-65 -> 3p3,4p1,5p4
## runs 66-70 -> 10 traces/key, 4p1 config
oceloteRun=0
pumaRun=1
typeOfStd="col"

trainSize=10
for numConfig in {3..3}
do
	for index in {66..70}
	do
		#Evaluates the configuration
	  jobName=run_dataEnsemble_colStd_"${index}"_3p3_4p1_5p4
	  echo "$jobName"
		if [[ $oceloteRun -eq 1 ]]
			then
			#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
	  	qsubCmd="qsub -N ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt -v modelName=${jobName},numPowerTraces=1500,numConfig=${numConfig},trainSize=${trainSize} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/run_aes_gem5_general_template_dataEnsemble.sh"
			echo "$qsubCmd"
			#Meval "$qsubCmd"

		elif [[ $pumaRun -eq 1 ]]
			then
			#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
		  sbatchCmd="sbatch --job-name ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt --export=modelName=${jobName},numPowerTraces=1500,numConfig=${numConfig},trainSize=${trainSize},typeOfStd=${typeOfStd} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/run_aes_gem5_general_template_dataEnsemble_puma.sh"
			echo "$sbatchCmd"
			eval "$sbatchCmd"
		fi
		
	done
done
