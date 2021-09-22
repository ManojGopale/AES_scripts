## Config4p1
## Runs 1-40 -> 28000 trainSize
## Runs 41-60 -> 15000 trainSize, cant run, will run this on 5p3 config instead
## 5p3_15000, runs 1-20
## 5p3_15000, runs 21-40 -> 3,4 hidden layers, 100-2000 nodes/layer
## 5p3_15000, runs 61-80 -> 1,2 hidden layers, 100-2000 nodes/layer
## runs 111-120 -> config 4p4, 28000 trainsize, random state = none in shuffle, 100-2000 nodes/HL
## runs 121-130 -> config4p4, 28000 trainSize, random state=none, 100-10000 node/HL
## runs 143, 144-> with tensorboard
## runs 145-150 -> with newLoadData function, 28000 trainSize, no tensorboard
## runs 151-160 -> with newLoadData, 15000 trainSize, with tensorboard
## 5/28/2021 -> Started testing 4.3 model by reducing NN complexity. 
## Config4p3
## runs 0-10 -> 1,2,3,4,5 HL, 32-512 nodes/HL
## runs 11-15 -> 10000 Training size
## runs 16-20 -> 5000 Training size
## runs40-45 -> 4p1 with 10traces/key and simpler model, to reduce overfitting ##Results are in dataEnsemble folder, changed after this run
## runs46-50 -> 4p1 with 50 traces/key, simpler models
## runs51-55 -> 4p1 with 100 traces/key, simpler models
## runs56-60 -> 4p1, 100 traces, row std, simpler models (did not do row std, used original traces)
## runs60-65 -> 4p1, 500 traces, row std, simpler models (did not do row std, used original traces)
## runs66-70 -> 4p1, 500 traces, row std, simpler models,  no shuffling of train data
## runs71-75 -> 4p1, 500 traces, row std, simpler models,  shuffling of train data before splitting
## runs76-80 -> 4p1, 1000 traces, row std, simpler models,  shuffling of train data before splitting
configList=(3p1 3p2 3p3 3p4 4p1 4p2 4p3 4p4 5p1 5p2 5p3 5p4 5p3_15000TrainSize 3p1_1361 3p2_1361)

oceloteRun=0
pumaRun=1
typeOfStd="row"

trainSize=1000
devSize=1000
for configIndex in {4..4}
do
	for index in {76..80}
	do
		#Evaluates the configuration
		jobName=run_config"${configList[${configIndex}]}"_"${index}"_"${trainSize}"tr_"${typeOfStd}"
		if [[ $oceloteRun -eq 1 ]]
			then
		  echo "$jobName"
			#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
		  qsubCmd="qsub -N ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt -v modelName=${jobName},numPowerTraces=1500,configName=config${configList[$configIndex]} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/run_aes_gem5_general_template.sh"
			echo "$qsubCmd"
			eval "$qsubCmd"
		elif [[ $pumaRun -eq 1 ]]
			then
	  	sbatchCmd="sbatch --job-name ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt --export=modelName=${jobName},numPowerTraces=1500,numConfig=${numConfig},trainSize=${trainSize},devSize=${devSize},configName=config${configList[$configIndex]},typeOfStd=${typeOfStd},combIndex=${index} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/run_aes_gem5_general_template_puma.sh"
			echo "$sbatchCmd"
			eval "$sbatchCmd"
		fi
		
	done
done
