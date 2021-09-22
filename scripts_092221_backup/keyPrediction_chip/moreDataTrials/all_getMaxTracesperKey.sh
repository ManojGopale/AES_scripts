interval=(10 150 151 152 153 154 155 156 157 158 159 160)
for index in {2..10}
do
	interval=${interval[$index]}
	#Evaluates the configuration
  jobName=getMaxTracesperKey_"${interval}"_15000TrainSize_169epochs_54p07
  echo "$jobName"
	#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
  qsubCmd="qsub -N ${jobName} -o ~/Chipwhiperer_Analysis/${jobName}_output.txt -e ~/Chipwhiperer_Analysis/${jobName}_error.txt -v interval=${interval} /xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/scr/getMaxTracesperKey_cw_template.sh"
	echo "$qsubCmd"
	eval "$qsubCmd"
done
