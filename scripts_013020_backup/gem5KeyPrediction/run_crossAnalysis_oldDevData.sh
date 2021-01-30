## This file runs the crossAnalysis for all the configurations test data for a particular model
configList=(config3p1 config3p2 config3p3 config3p4 config4p1 config4p2 config4p3 config4p4 config5p1 config5p2 config5p3 config5p4)

trainSize=28000
modelDir="/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/"
devDataDir="/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/"

## trainFlag=1, for training data
## trainFlag should be 1, since we are normalizing the data based on the trainData stats
trainFlag=1
## defFlag =1 when we want to train, or evaluate dev accuracy for hyper-paramter tuning
devFlag=1
## testFlag=1, when testing accuracy is required
testFlag=0
## The count of the config that needs to be tested
count=0

for index in {1..1}
do
	jobName=OldDevData_crossAnalysis_"${configList[${index}]}"_"${count}"
	##jobName=crossAnalysis_"${configList[${index}]}"
	echo "$jobName"

	#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
  qsubCmd="qsub -N ${jobName} -o ${modelDir}/${configList[${index}]}/${jobName}_output.txt -e ${modelDir}/${configList[${index}]}/${jobName}_error.txt -v devDataDir=${devDataDir},devDataConfig=${configList[${index}]},trainSize=${trainSize},modelDir=${modelDir},trainFlag=${trainFlag},devFlag=${devFlag},testFlag=${testFlag},count=${count} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/run_aes_crossAnalysis_template_oldDevData.sh"
	echo "$qsubCmd"
	eval "$qsubCmd"

done
