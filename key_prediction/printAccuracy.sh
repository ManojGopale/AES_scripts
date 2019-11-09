sizeList=(512 1024 2048 4096 6144 7168 8192 9216 10240 12288 16384)

for index in {0..11}
do
	path=/extra/manojgopale/AES_data/config3p1_15ktraining/batchSize_trials/size_"${sizeList[${index}]}"
	echo "$path"
	cmd="less $path/crossAnalysis_config3p1_config*_output.txt | egrep \"model\" | awk '{print \$NF}' "
	#echo $cmd
	eval $cmd
	echo ""
done

## Incorrect predictions count print
for index in {0..11}
do
	path=/extra/manojgopale/AES_data/config3p1_15ktraining/batchSize_trials/size_"${sizeList[${index}]}"
	echo "$path"
	cmd="less $path/crossAnalysis_config3p1_config*_output.txt | egrep \"incorrect prediction\" | awk '{print \$NF}' "
	#echo $cmd
	eval $cmd
	echo ""
done
