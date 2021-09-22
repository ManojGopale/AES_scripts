configList=(3p1 3p2 3p3 3p4 4p1 4p2 4p3 4p4 5p1 5p2 5p3 5p4 5p3_15000TrainSize 4p2_15000TrainSize 3p1_1361 3p2_1361)
for configIndex in {1..1}
do
	for index in {22..22}
	do
		#Evaluates the configuration
	  jobName=rerun_prev_config"${configList[$configIndex]}"_"${index}"_pool3p1_3p2
	  echo "$jobName"
		#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
	  sbatchCmd="sbatch --job-name ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt --export=modelName=${jobName},numPowerTraces=1500,configName=config${configList[$configIndex]} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/rerun_aes_gem5_general_template_puma.sh"
		echo "$sbatchCmd"
		eval "$sbatchCmd"
	done
done
