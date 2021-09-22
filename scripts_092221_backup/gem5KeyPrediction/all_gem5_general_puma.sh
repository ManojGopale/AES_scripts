## Config4p1
## Runs 1-40 -> 28000 trainSize
## Runs 41-60 -> 15000 trainSize, cant run, will run this on 5p3 config instead
## 5p3_15000, runs 1-20
## 5p3_15000, runs 21-40 -> 3,4 hidden layers, 100-2000 nodes/layer
## 5p3 15000, runs 41-60 -> ran in puma, 3-4 hidden layers, 100-2000 nodes/layer
configList=(3p1 3p2 3p3 3p4 4p1 4p2 4p3 4p4 5p1 5p2 5p3 5p4 5p3_15000TrainSize)
for configIndex in {12..12}
do
	for index in {41..60}
	do
		#Evaluates the configuration
	  jobName=run_config"${configList[${configIndex}]}"_"${index}"
	  echo "$jobName"
		#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
	  sbatchCmd="sbatch --job-name ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt --export=modelName=${jobName},numPowerTraces=1500,configName=config${configList[$configIndex]} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/run_aes_gem5_general_template_puma.sh"
		echo "$sbatchCmd"
		eval "$sbatchCmd"
	done
done
