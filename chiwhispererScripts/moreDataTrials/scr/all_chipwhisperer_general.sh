for index in {0..10}
do
	#Evaluates the configuration
  jobName=run_"${index}"
  echo "$jobName"
	#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
  qsubCmd="qsub -N ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt -v modelName=${jobName} /xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/scr/run_aes_chipWisperer_general_template.sh"
	echo "$qsubCmd"
	eval "$qsubCmd"
done
