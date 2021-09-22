echo "" > qdel_run.sh
for index in {10..54}
do
	cmd="qdel 2380${index}.elgato-adm"
	echo ${cmd} >> qdel_run.sh
done
