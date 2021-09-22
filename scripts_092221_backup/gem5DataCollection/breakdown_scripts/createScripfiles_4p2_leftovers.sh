### This file was created after elgato could not finish the last 3, 4 amd 5 in some cases
## keys from each set.
echo "" > config4p2_runScript_leftovers.sh
for keyIndex in {0..16}
	do
		if [[ $keyIndex -eq 0 ]]
			then
				startKey=11
			else
				## Adding 12 to each set because the last 3 are left in the set of 15
				startKey=$((keyIndex*15+1+11))
		fi
	endKey=$((($keyIndex+1)*15))
	cp -rf /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/breakdown_scripts/matlab_template_elgato_leftover_4p2.sh /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/breakdown_scripts/matlab_elgato_4p2_leftovers_${startKey}_${endKey}.sh 
	perl -pi -e "s/startKey/${startKey}/g" /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/breakdown_scripts/matlab_elgato_4p2_leftovers_${startKey}_${endKey}.sh
	perl -pi -e "s/endKey/${endKey}/g" /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/breakdown_scripts/matlab_elgato_4p2_leftovers_${startKey}_${endKey}.sh

	jobName=config4p2_gem5_to_mat_"${startKey}"_"${endKey}"_032121
	qsubCmd="qsub -N ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/breakdown_scripts/matlab_elgato_4p2_leftovers_${startKey}_${endKey}.sh"
	echo "$qsubCmd" >> config4p2_runScript_leftovers.sh
	echo "" >> config4p2_runScript_leftovers.sh
	echo "done"
done
