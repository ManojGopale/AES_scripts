for index in {177..177}
do
	#Evaluates the configuration
  jobName=rerun_"${index}"_2
  echo "$jobName"
	#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
  qsubCmd="qsub -N ${jobName} -o ~/Chipwhiperer_Analysis/${jobName}_output.txt -e ~/Chipwhiperer_Analysis/${jobName}_error.txt -v modelName=${jobName},numPowerTraces=1500 /xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/scr/rerun_aes_chipWisperer_general_template.sh"
	echo "$qsubCmd"
	eval "$qsubCmd"
done
