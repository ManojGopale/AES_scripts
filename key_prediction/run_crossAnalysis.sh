## This file runs the crossAnalysis for all the configurations test data for a particular model
configList=(config3p1 config3p2 config3p3 config3p4 config4p1 config4p2 config4p3 config4p4 config5p1 config5p2 config5p3 config5p4)

modelConfig=${configList[0]}
trainSize=15000
modelDir="/extra/manojgopale/AES_data/config3p1_15ktraining/dropOut_trials/size_1024_0p5_0p5_0p5/"
testDir="/xdisk/manojgopale/data_csv/"
modelName="dropOut_trial_3HLw_500_500_256_6epochs_Dropout_0p5_0p5_0p5_100p00.h5"
## trainFlag=1, for training data
trainFlag=0
## defFlag =1 when we want to train, or evaluate dev accuracy for hyper-paramter tuning
devFlag=1
## testFlag=1, when testing accuracy is required
testFlag=0

for index in {0..11}
do
	jobName=crossAnalysis_"${modelConfig}"_"${configList[${index}]}"
	echo "$jobName"

	#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
  qsubCmd="qsub -N ${jobName} -o ${modelDir}/${jobName}_output.txt -e ${modelDir}/${jobName}_error.txt -v modelConfig=${modelConfig},testConfig=${configList[${index}]},trainSize=${trainSize},modelDir=${modelDir},testDir=${testDir},modelName=${modelName},trainFlag=${trainFlag},devFlag=${devFlag},testFlag=${testFlag} /extra/manojgopale/AES_data/keyPrediction_scripts/run_aes_crossAnalysis_template.sh"
	echo "$qsubCmd"
	eval "$qsubCmd"

done
