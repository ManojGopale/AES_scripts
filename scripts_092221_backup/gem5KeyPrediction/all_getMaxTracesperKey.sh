interval=(10 100 200 300 400 500 600 700 800 900 1000)
for index in {0..10}
do
	interval=${interval[$index]}
	#Evaluates the configuration
  jobName=getMaxTracesperKey_5p3model_4p1devdata_"${interval}"_28000TrainSize
  echo "$jobName"
	#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
  qsubCmd="qsub -N ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt -v interval=${interval} /xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/scr/getMaxTracesperKey_cw_template.sh"
	echo "$qsubCmd"
	eval "$qsubCmd"
done
