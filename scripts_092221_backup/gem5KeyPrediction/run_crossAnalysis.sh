## This file runs the crossAnalysis for all the configurations test data for a particular model
configList=(config3p1 config3p2 config3p3 config3p4 config4p1 config4p2 config4p3 config4p4 config5p1 config5p2 config5p3 config5p4)

trainSize=28000
devSize=400
testSize=500
inputTraces=1500
modelDir="/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/"
devDataDir="/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/"
testDataDir="/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/"

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

for index in {0..11}
do
	##jobName=crossAnalysis_run22"${configList[${index}]}"_"${count}"
	jobName=interArchPortability_dataEnsemble_on"${configList[${index}]}"_testSize"${testSize}"
	##jobName=crossAnalysis_"${configList[${index}]}"
	echo "$jobName"

	if [[ $oceloteRun -eq 1 ]]
		then
		#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
  	qsubCmd="qsub -N ${jobName} -o ${modelDir}/${configList[${index}]}/${jobName}_output.txt -e ${modelDir}/${configList[${index}]}/${jobName}_error.txt -v devDataDir=${devDataDir},devDataConfig=${configList[${index}]},trainSize=${trainSize},modelDir=${modelDir},trainFlag=${trainFlag},devFlag=${devFlag},testFlag=${testFlag},count=${count} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/run_aes_crossAnalysis_template.sh"
		echo "$qsubCmd"
		eval "$qsubCmd"

	elif [[ $pumaRun -eq 1 ]]
		then
		#NOTE: Create a template for puma based on ocelote version
		## USed for hyper-paramter testing, with dev set
	  ##MsbatchCmd="sbatch --job-name ${jobName} -o ${modelDir}/${configList[${index}]}/${jobName}_output.txt -e ${modelDir}/${configList[${index}]}/${jobName}_error.txt --export=devDataDir=${devDataDir},devDataConfig=${configList[${index}]},trainSize=${trainSize},devSize=${devSize},modelDir=${modelDir},trainFlag=${trainFlag},devFlag=${devFlag},testFlag=${testFlag},count=${count},inputTraces=${inputTraces},typeOfStd=${typeOfStd} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/run_aes_crossAnalysis_template_puma.sh"

		## USed for final testing with test set
	  sbatchCmd="sbatch --job-name ${jobName} -o ${modelDir}/${configList[${index}]}/${jobName}_output.txt -e ${modelDir}/${configList[${index}]}/${jobName}_error.txt --export=testDataDir=${testDataDir},testDataConfig=${configList[${index}]},trainSize=${trainSize},testSize=${testSize},modelDir=${modelDir},trainFlag=${trainFlag},devFlag=${devFlag},testFlag=${testFlag},count=${count},inputTraces=${inputTraces},typeOfStd=${typeOfStd} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/run_aes_crossAnalysis_template_puma.sh"
		echo "$sbatchCmd"
		eval "$sbatchCmd"
	fi

done
