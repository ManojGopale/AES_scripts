##1. Load part of data
##2. Get the mean and std
##3. Load rest of the data
##4. Get mean and std


import pandas as pd
import numpy as np

import sys
sys.path.insert(0, '/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/')
import classify_general
import newLoadData
import time
import random
import gc
import numpy as np
import json

from optparse import OptionParser

parser = OptionParser()
parser.add_option('--trainSize',
									action = 'store', type='int', dest='trainSize', default = 28000)
parser.add_option('--resultDir',
									action = 'store', type='string', dest='resultDir', default = '/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/')
##Mparser.add_option('--modelName',
##M									action = 'store', type='string', dest='modelName', default = 'gem5Model')
parser.add_option('--configName',
									action = 'store', type='string', dest='configName', default = 'config3p1')
##Mparser.add_option('--trainFlag',
##M									action = 'store', type='int', dest='trainFlag', default = 1)
##Mparser.add_option('--devFlag',
##M									action = 'store', type='int', dest='devFlag', default = 1)
##Mparser.add_option('--testFlag',
##M									action = 'store', type='int', dest='testFlag', default = 0)
##Mparser.add_option('--numPowerTraces',
##M									action = 'store', type='int', dest='numPowerTraces', default = 1500)

(options, args) = parser.parse_args()

########
trainSize = options.trainSize
resultDir = options.resultDir
configName = options.configName
##MmodelName = options.modelName
##MtrainFlag = options.trainFlag
##MdevFlag = options.devFlag
##MtestFlag = options.testFlag
##MnumPowerTraces = options.numPowerTraces

dataDir = "/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/"
data=newLoadData.Data()
x_train, y_train = data.getData(dataDir, configName, trainSize, "Train")

##******************************************************************************##
## This section was to combine multiple mean and std. 
#MtotalTraces = x_train.shape[0]
#MhalfTraces = totalTraces/2
#Mprint("Get stats for 1st half of traces\n")
#Mx_train_mean, x_train_std = data.getStdParam(x_train.loc[:halfTraces-1,:])
#M## Write the mean and std to csv for future use
#MmeanPath = resultDir + "/" + configName + "/" + "mean1.csv"
#MstdPath = resultDir + "/" + configName + "/" + "std1.csv"
#Mx_train_mean.to_csv(meanPath, index=False, header=False)
#Mx_train_std.to_csv(stdPath, index=False, header=False)
#Mprint("Finihsed storing 1st half\n")
#M
#Mprint("Get stats for 2nd half of traces\n")
#Mx_train_mean, x_train_std = data.getStdParam(x_train.loc[halfTraces:,:])
#M## Write the mean and std to csv for future use
#MmeanPath = resultDir + "/" + configName + "/" + "mean2.csv"
#MstdPath = resultDir + "/" + configName + "/" + "std2.csv"
#Mx_train_mean.to_csv(meanPath, index=False, header=False)
#Mx_train_std.to_csv(stdPath, index=False, header=False)
#Mprint("Finihsed storing 2nd half\n")

##******************************************************************************##

## This section is used to get flat mean and std dev. for all the configurations

#MconfigStats= {}
#MstatPath = resultDir + "/" + configName + "/" + "fullStats_" + str(trainSize) + ".json"
#M
#Mflat_mean = np.mean(x_train.to_numpy())
#Mflat_std = np.std(x_train.to_numpy())
#M
#MconfigStats[configName] = {}
#MconfigStats[configName]["mean"] = flat_mean
#MconfigStats[configName]["stdDev"] = flat_std
#M
#Mjson.dump(configStats, open(statPath, 'w'))

##******************************************************************************##
## This section will use 100 traces/key/config and collate all the datasets. Then get the mean and stdDev for them.
## This should make the outliers with small dev go away hopefully.

configList = ["config3p1", "config3p2", "config3p3", "config3p4", "config4p1", "config4p2", "config4p3", "config4p4", "config5p1", "config5p2", "config5p3", "config5p4"]

dataDir = "/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/"
data=newLoadData.Data()

x_train = pd.DataFrame()
trainSize=100
typeofData = "Train"

for index, config in enumerate(configList):
	x_inter, _ = data.getData(dataDir, config, trainSize, typeofData)
	x_train = pd.concat([x_train, x_inter], ignore_index=True)

print("After concatenation shapes are x_train= %s, y_train= %s\n" %(x_train.shape, y_train.shape))

#MstatPath = resultDir + "/allConfigs/fullStats_" + typeofData + "_" + str(trainSize) + ".json"
#Mflat_mean = np.mean(x_train.to_numpy())
#Mflat_std = np.std(x_train.to_numpy())
#M
#MconfigStats[configName] = {}
#MconfigStats[configName]["mean"] = flat_mean
#MconfigStats[configName]["stdDev"] = flat_std
#M
#Mjson.dump(configStats, open(statPath, 'w'))

## Get columnwise mean and stdDev
x_col_mean, x_col_std = data.getStdParam(x_train)
## Write the mean and std to csv for future use
meanPath = resultDir + "/allConfigs/fullStats_" + typeofData + "_" +"mean.csv"
stdPath = resultDir + "/allConfigs/fullStats_" + typeofData + "_" +"std.csv"
x_col_mean.to_csv(meanPath, index=False, header=False)
x_col_std.to_csv(stdPath, index=False, header=False)
##******************************************************************************##
