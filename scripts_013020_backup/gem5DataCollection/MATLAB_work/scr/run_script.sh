module load matlab/r2018a
matlab ##Invoking the shell
cd /xdisk/rlysecky/manojgopale/extra/gem5DataCollection/scr/MATLAB_work/scr/ ##Going to directory where all the .m files are located

#Inputs to script
#qResult_folders = ["2019-06-11_21-17-45default_cache", "2019-06-11_21-17-52default_cache", "2019-06-11_21-17-59default_cache"];
qResult_folders = ["2020-11-13_05-40-41default_cache"];
#outputPath = "/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/2019-06-11_21-17-45_52_59_default_cache/";
outputPath = "/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/matResult/2020-11-13_05-40-41default_cache/";

#Generate power traces for the configuration
generateTrace(qResult_folders, outputPath);
