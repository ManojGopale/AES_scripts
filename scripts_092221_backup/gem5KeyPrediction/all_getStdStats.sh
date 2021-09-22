configList=(3p1 3p2 3p3 3p4 4p1 4p2 4p3 4p4 5p1 5p2 5p3 5p4 5p3_15000TrainSize 3p1_1361 3p2_1361)
oceloteRun=0
pumaRun=1
trainSize=100
for configIndex in {1..1}
do
	#Evaluates the configuration
  #jobName=getStdStats_config"${configList[${configIndex}]}"
  jobName=getStdStats_allConfig_tr_"${trainSize}"_col
  echo "$jobName"
	if [[ $oceloteRun -eq 1 ]]
		then
		#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
	  qsubCmd="qsub -N ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt -v configName=config${configList[$configIndex]} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/run_stdStats_template.sh"
		echo "$qsubCmd"
		eval "$qsubCmd"

	elif [[ $pumaRun -eq 1 ]]
		then
		#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
	  sbatchCmd="sbatch --job-name ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt --export=configName=config${configList[$configIndex]},trainSize=${trainSize} /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/run_stdStats_template_puma.sh"
		echo "$sbatchCmd"
		eval "$sbatchCmd"
	fi
done

