import sys
sys.path.insert(0, '/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/')
import classify_general
import time
import numpy as np
import pandas as pd

import scipy.io as si
import numpy as np
import argparse
import pickle
import gzip
import time
from optparse import OptionParser
import random


##MconfigName = "config4p1"
##MtrainSize = 28000
##MtrainFlag = 1
##MdevFlag = 1
##MtestFlag = 0
##M
##MdataDir = "/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/"
##MtrainData, devData, _ = classify_general.getData(dataDir, configName, trainSize, trainFlag, devFlag, testFlag)
##M
##Mx_train, y_train_oh = trainData
##Mx_dev, y_dev_oh = devData

## We can also directly load the csv files,
csvFilePath="/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/config4p1/aesData_config9_Train_0.csv"
train=pd.read_csv(csvFilePath, header=None)

## Get data for key==11.
key = 11 ## Can change the key too
train_11 = train.loc[train.iloc[:,1500] == 11].iloc[:,:-1]

## get matlab data to compare against
parentDir = "/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/"
## matConfig can change for different matresults
matConfig = "config3p1"
key = 11 ## Can change the key too
keyStr = parentDir  + '/matResult/' + matConfig + '/value' + str(key) + '.mat'
keyData = si.loadmat(keyStr)
keyDataArray = keyData['power'].reshape(keyData['power'].shape)[:,0:1500]

mat_11 = pd.DataFrame(data=keyDataArray)
con4p1_mat_3p1_11 = pd.concat([mat_11,train_11])
print("Shape before finding duplicates = %s\nAfter dropping dup0licates= %s" %(con4p1_mat_3p1_11.shape,con4p1_mat_3p1_11.drop_duplicates().shape))

matConfig = "config5p1"
key=11
keyStr = parentDir  + '/matResult/' + matConfig + '/value' + str(key) + '.mat'
keyData = si.loadmat(keyStr)
keyDataArray = keyData['power'].reshape(keyData['power'].shape)[:,0:1500]

mat_5p1_11 = pd.DataFrame(data=keyDataArray)
mat_5p1_3p1_11 = pd.concat([mat_11, mat_5p1_11])

print("Shape before finding duplicates = %s\nAfter dropping dup0licates= %s" %(mat_5p1_3p1_11.shape,mat_5p1_3p1_11.drop_duplicates().shape))

