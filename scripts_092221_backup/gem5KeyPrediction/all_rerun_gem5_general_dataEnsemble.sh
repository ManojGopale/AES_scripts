##configList=(3p1 3p2 3p3 3p4 4p1 4p2 4p3 4p4 5p1 5p2 5p3 5p4 5p3_15000TrainSize)
## reruns 1-10 -> 2 configs, with run22 param
## with Standardize class and random_state=none
## reruns 11-20 -> Standard class, random_state=none in shuffle, also std after combining data
## rerun 12- 14-> no std function (mem error), random state = none
## rerun49_0_10 -> 10 runs with 3 configs with rerun49 arch
## rerun49_11_20 -> 10 runs with 3 configs with rerun49 arch, trainSize=2000traces/key/config
## rerun49_21_30 -> 10 runs with 2 configs with rerun49 arch, trainSize=1000traces/key/config
## rerun49_31_45 -> Taking data of config_3p3_5p4_4p3 we try to tune the DNN model to get better
## combTrials_[0-19] -> Combination trials of all 20 triplets of p1+p4's with rerun49_43 model.
## combTrials_20 -> 4p1+5p4 dataset, model srerun49_43
## combTrials_21,22 -> 4p1, 5p4 dataset, model srerun49_43
## combTrials_2_[0-5] -> Trails on dataset 2, with random seed to see if that changes the low performance
## rerun49_5_dataEnsemble_sigmoidTrial -> Using rerun49_5 as the base model and changing activations to sigmoid
## rerun54_4p1 -> 4p1 rerun model51 with no batch norm for freeze layer trials
## rerun54_4p2 -> 4p2 rerun model51 with no batch norm for freeze layer trials

##rerun43_49_S7_trSize[trainingSize] -> training size trials 10, 20, 30, 50, 100, 200, 500, 1000, S7->config4p1,5p1,5p4

oceloteRun=0
pumaRun=1
typeOfStd="col"

trainSize=1250
devSize=500
configName="dataEnsemble"
for numConfig in {3..3}
do
	for index in {0..0}
	do
		#Evaluates the configuration
	  #jobName=rerun49_43_"${index}"_dataEnsemble_3p3_5p4_4p3_NNTrials
	  jobName=rerun49_43_S12_"${trainSize}"
	  #jobName=rerun49_config4p4
	  echo "$jobName"
		if [[ $oceloteRun -eq 1 ]]
			then
			#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
	  	qsubCmd="qsub -N ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt -v modelName=${jobName},numPowerTraces=1500,numConfig=${numConfig},trainSize=${trainSize} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/rerun_aes_gem5_general_template_dataEnsemble.sh"
			echo "$qsubCmd"
			eval "$qsubCmd"

		elif [[ $pumaRun -eq 1 ]]
			then
	  	sbatchCmd="sbatch --job-name ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt --export=modelName=${jobName},numPowerTraces=1500,numConfig=${numConfig},trainSize=${trainSize},typeOfStd=${typeOfStd},combIndex=${index},devSize=${devSize},configName=${configName} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/rerun_aes_gem5_general_template_dataEnsemble_puma.sh"
			echo "$sbatchCmd"
			eval "$sbatchCmd"
		fi
		
	done
done
