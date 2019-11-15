## This will get the accuracu=y of a given configuration against all other 
## configuration test data, also will perform preliminary error analysis 

## Given modelDir, csv files will be saved in those directory.

import pandas as pd
import numpy as np
from keras.models import load_model

import sys
sys.path.insert(0, '/extra/manojgopale/AES_data/keyPrediction_scripts/')
import classify_3HL
from error_analysis import errorAnalysis

from optparse import OptionParser

parser = OptionParser()
parser.add_option('-m','--modelConfig',
									action = 'store', type='string', dest='modelConfig', default = 'config3p1')
parser.add_option('-t','--testConfig',
									action = 'store', type='string', dest='testConfig', default = 'config3p1')
parser.add_option('--trainSize',
									action = 'store', type='int', dest='trainSize', default = 15000)
parser.add_option('--modelDir',
									action = 'store', type='string', dest='modelDir', default = '/extra/manojgopale/AES_data/config3p1_15ktraining/batchSize_trials/size_16000/')
parser.add_option('--testDir',
									action = 'store', type='string', dest='testDir', default = '/xdisk/manojgopale/data_csv/')
parser.add_option('--modelName',
									action = 'store', type='string', dest='modelName', default = 'model_batchSize_3HLw_500_500_256_noDrop_10epochs_0p2Dropout_99p52.h5')
parser.add_option('--testFlag',
									action = 'store', type='int', dest='testFlag', default = 0)

(options, args) = parser.parse_args()

########

modelConfig = options.modelConfig
testConfig = options.testConfig
trainSize = options.trainSize
modelDir = options.modelDir
testDir = options.testDir
modelName = options.modelName
testFlag = options.testFlag

########

modelPath = modelDir + "/" + modelName
model = load_model(modelPath)

testDataPath = testDir + "/" + testConfig + "/"
_, _, testData = classify_3HL.getData(testDataPath, trainSize, testFlag)
x_test, y_test_oh = testData

## Evaluate the performance of model on testData
model_score = model.evaluate(x_test, y_test_oh, batch_size=2048)
print("\nmodel= %s score on %s is: %s\n" %(modelConfig, testConfig, model_score[1]))

## Convert from one-hot to numerical prediction
y_pred = np.argmax(model.predict(x_test, batch_size=2048), axis=1)

## vstack the actual and predicted output and take transpose
output_predict = np.vstack((np.argmax(y_test_oh, axis=1), y_pred)).T

## Save it to csv file for future analysis
outputFile = modelDir + "/"  + modelConfig + "_" + testConfig + ".csv" 
np.savetxt(outputFile, output_predict, fmt="%5.0f", delimiter=",")

##Error Analysis
errorAnalysis(outputFile)

print("#########--------------###########\n")
