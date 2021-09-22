## totalRun
totalRun=(3 3 3 3 3 3 3 3 3 3 3 3)

##configName
configName=(3p1 3p2 3p3 3p4 4p1 4p2 4p3 4p4 5p1 5p2 5p3 5p4)

##cpuCount ocellato
cpuCount=(28 28 28 28 28 28 28 28)

for index in {9..9}
do
	mkCmd="mkdir -p /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config${configName[${index}]}"
	echo $mkCmd
	#Meval $mkCmd
	for keyIndex in {13..16}
	do
		if [[ $keyIndex -eq 0 ]]
			then
				startKey=0
			else
				startKey=$((keyIndex*15+1))
		fi

		#Evaluates the configuration
	  jobName=config"${configName[${index}]}"_gem5_to_mat_"$startKey"_1120
	  echo "$jobName"
		#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
		endKey=$((($keyIndex+1)*15))
		#### Ocelote
	  qsubCmd="qsub -N ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt -v startKey=${startKey},endKey=${endKey},configName=config${configName[${index}]} /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/matlab_template.sh"
		echo "$qsubCmd"
		eval "$qsubCmd"

		#### ELGato
	  #MqsubCmd="qsub -N ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt -v startKey=${startKey},endKey=${endKey},configName=config${configName[${index}]} /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/matlab_template_elgato.sh"
		#Mecho "$qsubCmd"
		#Meval "$qsubCmd"

		#### Puma
	  #MsbatchCmd="sbatch --job-name ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt --export=startKey=${startKey},endKey=${endKey},configName=config${configName[${index}]} /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/matlab_template_parfeval.sh"
		#Mecho "$sbatchCmd"
		#Meval "$sbatchCmd"
	done
done
