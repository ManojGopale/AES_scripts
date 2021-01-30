##configName
configName=(3p1 3p2 3p3 3p4 4p1 4p2 4p3 4p4 5p1 5p2 5p3 5p4)

for index in {9..9}
do
	mkCmd="mkdir -p /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/config${configName[${index}]}"
	echo $mkCmd
	eval $mkCmd
	#Evaluates the configuration
  jobName=config"${configName[${index}]}"_mat_to_csv_1120
  echo "$jobName"
	#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
  qsubCmd="qsub -N ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt -v configName=config${configName[${index}]} /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/csv_template.sh"
	echo "$qsubCmd"
	eval "$qsubCmd"
done
