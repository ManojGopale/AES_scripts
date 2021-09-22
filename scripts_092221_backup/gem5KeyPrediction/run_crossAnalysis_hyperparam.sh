## This file runs the crossAnalysis for all the configurations test data for a particular model
configList=(config3p1 config3p2 config3p3 config3p4 config4p1 config4p2 config4p3 config4p4 config5p1 config5p2 config5p3 config5p4 config5p3_15000TrainSize config4p2_15000TrainSize config3p1_1361 config3p2_1361)

## NOTE: change trainSize if required
trainSize=15000
devSize=400
testSize=500
inputTraces=1500
modelDir="/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/"
devDataDir="/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/"
testDataDir="/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/"
## config whose model we are loading
modelConfig="dataEnsemble"
## Type of standardization that was applied while training
typeOfStd="col"
## CLuster to choose from
oceloteRun=0
pumaRun=1

## trainFlag=1, for training data
## trainFlag should be 1, since we are normalizing the data based on the trainData stats
trainFlag=1
## defFlag =1 when we want to train, or evaluate dev accuracy for hyper-paramter tuning
devFlag=1
## testFlag=1, when testing accuracy is required
testFlag=0

## index of the config on whom we want to test the models, different than modelConfig
for index in {0..0}
do
	jobName=crossAnalysis_"${typeOfStd}"_"${modelConfig}"_"${configList[${index}]}"_"${devSize}"devSize_"${inputTraces}"_S7
	##jobName=crossAnalysis_"${configList[${index}]}"
	echo "$jobName"

	if [[ $oceloteRun -eq 1 ]]
		then
	#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
	  qsubCmd="qsub -N ${jobName} -o ${modelDir}/${configList[${index}]}/${jobName}_output.txt -e ${modelDir}/${configList[${index}]}/${jobName}_error.txt -v devDataDir=${devDataDir},devDataConfig=${configList[${index}]},trainSize=${trainSize},devSize=${devSize},modelDir=${modelDir},trainFlag=${trainFlag},devFlag=${devFlag},testFlag=${testFlag},count=${count},modelConfig=${modelConfig},inputTraces=${inputTraces},typeOfStd=${typeOfStd} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/run_aes_crossAnalysis_hyperparam_template.sh"
		echo "$qsubCmd"
		eval "$qsubCmd"

	elif [[ $pumaRun -eq 1 ]]
		then
	  sbatchCmd="sbatch --job-name ${jobName} -o ${modelDir}/${configList[${index}]}/${jobName}_output.txt -e ${modelDir}/${configList[${index}]}/${jobName}_error.txt --export=devDataDir=${devDataDir},devDataConfig=${configList[${index}]},trainSize=${trainSize},devSize=${devSize},modelDir=${modelDir},trainFlag=${trainFlag},devFlag=${devFlag},testFlag=${testFlag},count=${count},modelConfig=${modelConfig},inputTraces=${inputTraces},typeOfStd=${typeOfStd} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/run_aes_crossAnalysis_hyperparam_template_puma.sh"
		echo "$sbatchCmd"
		eval "$sbatchCmd"
	fi

done
