configList=(3p1 3p2 3p3 3p4 4p1 4p2 4p3 4p4 5p1 5p2 5p3 5p4)
for configIndex in {9..9}
do
	for index in {177..177}
	do
		#Evaluates the configuration
	  jobName=rerun_cw_run_config"${configList[$configIndex]}"_"${index}"
	  echo "$jobName"
		#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
	  qsubCmd="qsub -N ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt -v modelName=${jobName},numPowerTraces=1500,configName=config${configList[$configIndex]} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/rerun_aes_gem5_general_template.sh"
		echo "$qsubCmd"
		eval "$qsubCmd"
	done
done
