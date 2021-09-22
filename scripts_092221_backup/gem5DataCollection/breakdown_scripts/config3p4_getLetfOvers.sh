## The first portion is to get how many are reaminingin each set of 10 numbers
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value1[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value2[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value3[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value4[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value5[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value6[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value7[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value8[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value9[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value10[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value11[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value12[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value13[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value14[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value15[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value16[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value17[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value18[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value19[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value20[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value21[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value22[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value23[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value24[0-9].mat  | wc -l`
###Ma=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value25[0-9].mat  | wc -l`
###M
for index in {0..25}
do
	if [[ ${index} -eq 0 ]]
		then
			a=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value[0-9].mat  | wc -l`
			echo "${index}= $a"
	else
			a=`l /xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/config3p4/value${index}[0-9].mat  | wc -l`
			echo "${index}= $a"
		fi
done

## This part is to create file that can be sources in puma
###Mecho "" > config3p4_runScript_leftovers.sh
###M## Get all the manually modified files that need to be run
###M## https://stackoverflow.com/questions/15065010/how-to-perform-a-for-each-loop-over-all-the-files-under-a-specified-path
###Mfind /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/breakdown_scripts -name "matlab_puma_3p4_leftovers*" -type f -print0 | while IFS= read -r -d $'\0' line
###Mdo
###M	keysLeft=`l $line | cut -d "/" -f9 | cut -d "." -f1 | cut -d "_" -f5-6`
###M	echo "${keysLeft}"
###M	jobName=config3p4_gem5_to_mat_leftovers_"${keysLeft}"_031821
###M	sbatchCmd="sbatch --job-name ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt ${line}"
###M	echo "$sbatchCmd" >> config3p4_runScript_leftovers.sh
###M	echo "" >> config3p4_runScript_leftovers.sh
###M	echo "done"
###Mdone
