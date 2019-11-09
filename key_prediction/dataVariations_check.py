import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '/extra/manojgopale/AES_data/keyPrediction_scripts/')
import classify_3HL 
import time

data_3p1_dir= "/xdisk/manojgopale/data_csv/config3p1/"
data_5p2_dir= "/xdisk/manojgopale/data_csv/config5p2/"
trainSize = 15000
testFlag = 0

## Getting data for 3p1 and 5p2
trainData_3p1, devData_3p1, testData_3p1 = classify_3HL.getData(data_3p1_dir, trainSize, testFlag)

## Since train and dev data is not avaialable we are using test
## Change testFlag to 0

##trainData_5p2, devData_5p2, testData_5p2 = classify_3HL.getData(data_5p2_dir, trainSize, testFlag)
trainData_5p2, devData_5p2, testData_5p2 = classify_3HL.getData(data_5p2_dir, trainSize, 1)
