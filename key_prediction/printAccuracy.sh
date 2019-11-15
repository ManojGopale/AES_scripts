## Batchsize trials
#sizeList=(64 128 256 512 1024 2048 4096 6144 7168 8192 9216 10240 12288 16384)
sizeList=(64 128 256 512 1024 2048 4096 8192 10240 16384)

## DropOut trials
#sizeList=(size_1024_0p1_0p1_0p1 size_1024_0p3_0p3_0p3 size_1024_0p4_0p4_0p4 size_1024_0p5_0p5_0p5)

## Across configurations
configList=(config3p2 config3p3 config3p4 config4p1 config4p2 config4p3 config4p4 config5p1 config5p2 config5p3 config5p4)
trainDir=(batch_1024_0p2_0p2_0p2 batch_1024_0p3_0p3_0p3)

for index in {0..9}
do
	## BatchSize Trials
	path=/extra/manojgopale/AES_data/config3p1_15ktraining/batchSize_trials/size_"${sizeList[${index}]}"
	echo "$path"
	cmd="less $path/crossAnalysis_config3p1_config*_output.txt | egrep \"model\" | awk '{print \$NF}'"
	#echo "$cmd"
	eval "$cmd"
	echo ""

	##Print Accuracy across configuration for test runs
#M	## DropOut Trials
#M	path_0p2=/extra/manojgopale/AES_data/"${configList[${index}]}"_15ktraining/trainRuns/"${trainDir[0]}"
#M	path_0p3=/extra/manojgopale/AES_data/"${configList[${index}]}"_15ktraining/trainRuns/"${trainDir[1]}"
#M	echo "$path_0p2"
#M	cmd_0p2="less $path_0p2/crossAnalysis_config3p1_config*_output.txt | egrep \"model\" | awk '{print \$NF *100}' "
#M	#cmd_0p2="less $path_0p2/crossAnalysis_config3p1_config*_output.txt | egrep \"model\" "
#M	#echo $cmd_0p2
#M	eval $cmd_0p2
#M
#M	echo "$path_0p3"
#M	cmd_0p3="less $path_0p3/crossAnalysis_config3p1_config*_output.txt | egrep \"model\" | awk '{print \$NF*100}' "
#M	#cmd_0p3="less $path_0p3/crossAnalysis_config3p1_config*_output.txt | egrep \"model\" "
#M	#echo $cmd_0p3
#M	eval $cmd_0p3
#M	echo ""
done

## Incorrect predictions count print
for index in {0..10}
do
	## BatchSize Trials
	path=/extra/manojgopale/AES_data/config3p1_15ktraining/batchSize_trials/size_"${sizeList[${index}]}"
	echo "$path"
	cmd="less $path/crossAnalysis_config3p1_config*_output.txt | egrep \"incorrect prediction\" | awk '{print \$NF}'"
	#echo "$cmd"
	eval "$cmd"
	echo ""

	##Print Accuracy across configuration for test runs

#M	## DropOut Trials
#M	path_0p2=/extra/manojgopale/AES_data/"${configList[${index}]}"_15ktraining/trainRuns/"${trainDir[0]}"
#M	path_0p3=/extra/manojgopale/AES_data/"${configList[${index}]}"_15ktraining/trainRuns/"${trainDir[1]}"
#M	echo "$path_0p2"
#M	cmd_0p2="less $path_0p2/crossAnalysis_config3p1_config*_output.txt | egrep \"incorrect prediction\" | awk '{print \$NF}' "
#M	#echo $cmd_0p2
#M	eval $cmd_0p2
#M
#M	echo "$path_0p3"
#M	cmd_0p3="less $path_0p3/crossAnalysis_config3p1_config*_output.txt | egrep \"incorrect prediction\" | awk '{print \$NF}' "
#M	#echo $cmd_0p3
#M	eval $cmd_0p3
#M	echo ""
done
