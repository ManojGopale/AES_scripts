config=(config3p1 config3p2 config3p3 config3p4 config4p1 config4p2 config4p3 config4p4 config5p1 config5p2 config5p3 config5p4)

for configIndex in {10..10}
do
	for runIndex in {21..40}
	do
		#Evaluates the configuration
	  jobName="${config[${configIndex}]}"_"${runIndex}"
	  echo "$jobName"
		#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
	  qsubCmd="qsub -N ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt -v modelName=${jobName},numPowerTraces=1361,config=${config[${configIndex}]} /xdisk/rlysecky/manojgopale/extra/gem5Trails/scr/rerun_aes_chipWisperer_general_template.sh"
		echo "$qsubCmd"
		eval "$qsubCmd"
	done
done
