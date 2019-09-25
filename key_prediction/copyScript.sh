#configList="3p1 3p2 3p3 3p4 4p1 4p2 4p3 4p4 5p1 5p2 5p3 5p4"
configList="3p2 3p3 3p4 4p1 4p2 4p3 4p4 5p1 5p2 5p3 5p4"

for config in $configList
do
  echo "$config"
  ## Copy template from 3p1 to other configs
  copyCmd="cp -rf /extra/manojgopale/AES_data/config3p1_15ktraining/scr/run_2hl.py /extra/manojgopale/AES_data/config${config}_15ktraining/scr/."
  echo "$copyCmd"
  eval $copyCmd

  ## Substitute 3p1 to respective configNum
  replaceCmd="perl -pi -e 's/3p1/${config}/g' /extra/manojgopale/AES_data/config${config}_15ktraining/scr/run_2hl.py"
  echo "$replaceCmd"
  eval $replaceCmd
  echo ""
done

