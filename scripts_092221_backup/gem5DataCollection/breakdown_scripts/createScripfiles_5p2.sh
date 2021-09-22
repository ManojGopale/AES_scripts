echo "" > config5p2_runScript.sh
for keyIndex in {0..16}
	do
		if [[ $keyIndex -eq 0 ]]
			then
				startKey=0
			else
				startKey=$((keyIndex*15+1))
		fi
	endKey=$((($keyIndex+1)*15))
	cp -rf /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/breakdown_scripts/matlab_template_ocelote_5p2.sh /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/breakdown_scripts/matlab_ocelote_5p2_${startKey}_${endKey}.sh 
	perl -pi -e "s/startKey/${startKey}/g" /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/breakdown_scripts/matlab_ocelote_5p2_${startKey}_${endKey}.sh
	perl -pi -e "s/endKey/${endKey}/g" /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/breakdown_scripts/matlab_ocelote_5p2_${startKey}_${endKey}.sh

	jobName=config5p2_gem5_to_mat_"${startKey}"_"${endKey}"_040421
	qsubCmd="qsub -N ${jobName} -o ~/GEM5_Analysis/${jobName}_output.txt -e ~/GEM5_Analysis/${jobName}_error.txt /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/breakdown_scripts/matlab_ocelote_5p2_${startKey}_${endKey}.sh"
	echo "$qsubCmd" >> config5p2_runScript.sh
	echo "" >> config5p2_runScript.sh
	echo "done"
done
