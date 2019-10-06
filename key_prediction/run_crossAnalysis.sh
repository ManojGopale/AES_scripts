## This file runs the crossAnalysis for all the configurations test data for a particular model
configList=(config3p1 config3p2 config3p3 config3p4 config4p1 config4p2 config4p3 config4p4 config5p1 config5p2 config5p3 config5p4)

modelConfig=${configList[0]}
trainSize=15000
modelDir="/extra/manojgopale/AES_data/config3p1_15ktraining/batchSize_trials/size_2048/"
testDir="/xdisk/manojgopale/data_csv/"
modelName="model_batchSize_3HLw_500_500_256_noDrop_12epochs_0p2Dropout_98p42.h5"


for index in {8..8}
do
	jobName=crossAnalysis_"${modelConfig}"_"${configList[${index}]}"
	echo "$jobName"

	#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
  qsubCmd="qsub -N ${jobName} -o ~/AES_error_output_dir/${jobName}_output.txt -e ~/AES_error_output_dir/${jobName}_error.txt -v modelConfig=${modelConfig},testConfig=${configList[${index}]},trainSize=${trainSize},modelDir=${modelDir},testDir=${testDir},modelName=${modelName} /extra/manojgopale/AES_data/keyPrediction_scripts/run_aes_crossAnalysis_template.sh"
	echo "$qsubCmd"
	eval "$qsubCmd"

done
