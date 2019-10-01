## totalRun
totalRun=(1 1 1 1 1 1 1 1)

##configName
configName=(default_cache cache_associate8 64kcache 64k_associate8 l2default l2_l1associate8 l24m_default l24m_l1associate8)

##cpuCount ocellato
cpuCount=(28 28 28 28 28 28 28 28)

for index in {1..1}
do
	#Evaluates the configuration
  jobName="${configName[${index}]}"_fib_500_WK
  echo "$jobName"
	#--- Make sure to remove -o and -e from ~/run_aes_template.sh ---###
  qsubCmd="qsub -N ${jobName} -o ~/GEM5_Analysis/${jobName}_1_output.txt -e ~/GEM5_Analysis/${jobName}_1_error.txt -v totalRun=${totalRun[${index}]},configName=${configName[${index}]},cpuCount=${cpuCount[${index}]} ~/gem5_template.sh"
	echo "$qsubCmd"
	eval "$qsubCmd"
done
