## This file runs the crossAnalysis for all the configurations test data for a particular model
configList=(config3p1 config3p2 config3p3 config3p4 config4p1 config4p2 config4p3 config4p4 config5p1 config5p2 config5p3 config5p4)

trainSize=15000
modelDir="/xdisk/rlysecky/manojgopale/extra/gem5Trails/result/"
testDir="/xdisk/rlysecky/manojgopale/extra/gem5Trails/data/"

## trainFlag=1, for training data
trainFlag=0
## defFlag =1 when we want to train, or evaluate dev accuracy for hyper-paramter tuning
devFlag=1
## testFlag=1, when testing accuracy is required
testFlag=0
## The count of the config that needs to be tested
count=8

for index in {8..11}
do
	jobName=crossAnalysis_"${configList[${index}]}"_"${count}"
	echo "$jobName"

	#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
  qsubCmd="qsub -N ${jobName} -o ${modelDir}/${configList[${index}]}/${jobName}_output.txt -e ${modelDir}/${configList[${index}]}/${jobName}_error.txt -v testConfig=${configList[${index}]},trainSize=${trainSize},modelDir=${modelDir},testDir=${testDir},trainFlag=${trainFlag},devFlag=${devFlag},testFlag=${testFlag},count=${count} /xdisk/rlysecky/manojgopale/extra/gem5Trails/scr/run_aes_crossAnalysis_template.sh"
	echo "$qsubCmd"
	eval "$qsubCmd"

done
