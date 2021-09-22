## totalRun
totalRun=(3 3 3 3 3 3 3 3 3 3 3 3)

##configName
configName=(default_cache cache_associate8 64kcache 64k_associate8 l2default l2_l1associate8 l24m_default l24m_l1associate8 dram_l2default dram_l2_l1associate8 dram_l24m_default dram_l24m_l1associate8)

##cpuCount ocellato
cpuCount=(28 28 28 28 28 28 28 28 28 28 28 28)

for index in {0..11}
do
	#Evaluates the configuration
  jobName="${configName[${index}]}"_gem5Data_030120
  echo "$jobName"
	#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
  qsubCmd="qsub -N ${jobName} -o ~/GEM5_Analysis/${jobName}_1_output.txt -e ~/GEM5_Analysis/${jobName}_1_error.txt -v totalRun=${totalRun[${index}]},configName=${configName[${index}]},cpuCount=${cpuCount[${index}]} /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/gem5_template.sh"
	echo "$qsubCmd"
	eval "$qsubCmd"
done
