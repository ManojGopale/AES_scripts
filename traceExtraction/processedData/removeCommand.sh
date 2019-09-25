ls -ld *  | egrep -v "mainFiles" | awk '{print $9}' | xargs rm -rf
