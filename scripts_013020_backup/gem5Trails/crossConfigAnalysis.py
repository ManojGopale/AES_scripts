## This will get the accuracu=y of a given configuration against all other 
## configuration test data, also will perform preliminary error analysis 

## Given modelDir, csv files will be saved in those directory.

import pandas as pd
import numpy as np
from keras.models import load_model

import sys
sys.path.insert(0, '/xdisk/rlysecky/manojgopale/extra/gem5Trails/scr/')
import classify_general
from error_analysis import errorAnalysis

from optparse import OptionParser

parser = OptionParser()
parser.add_option('-t','--testConfig',
									action = 'store', type='string', dest='testConfig', default = 'config3p1')
parser.add_option('--trainSize',
									action = 'store', type='int', dest='trainSize', default = 15000)
parser.add_option('--modelDir',
									action = 'store', type='string', dest='modelDir', default = '/xdisk/rlysecky/manojgopale/extra/gem5Trails/result/')
parser.add_option('--testDir',
									action = 'store', type='string', dest='testDir', default = '/xdisk/rlysecky/manojgopale/extra/gem5Trails/data/')
## modelList will include all the best DNN models for which we want to test
parser.add_option('--trainFlag',
									action = 'store', type='int', dest='trainFlag', default = 0)
parser.add_option('--devFlag',
									action = 'store', type='int', dest='devFlag', default = 0)
parser.add_option('--testFlag',
									action = 'store', type='int', dest='testFlag', default = 0)
parser.add_option('--count',
									action = 'store', type='int', dest='count', default = 0)

(options, args) = parser.parse_args()

########

testConfig  = options.testConfig
trainSize   = options.trainSize
modelDir 		= options.modelDir
testDir 		= options.testDir
trainFlag 	= options.trainFlag
devFlag 		= options.devFlag
testFlag 		= options.testFlag
count 		= options.count

########
configList = ["config3p1", "config3p2", "config3p3", "config3p4", "config4p1", "config4p2", "config4p3", "config4p4","config5p1", "config5p2", "config5p3", "config5p4"]

modelList=["config3p1_1_4HL_8epochs_97p86_acc_.h5", "config3p2_1_4HL_7epochs_99p99_acc_.h5", "config3p3_1_4HL_6epochs_99p98_acc_.h5", "config3p4_1_4HL_7epochs_99p97_acc_.h5", "config4p1_1_4HL_12epochs_99p99_acc_.h5", "config4p2_1_4HL_11epochs_100p00_acc_.h5", "config4p3_1_4HL_7epochs_99p99_acc_.h5", "config4p4_1_4HL_8epochs_99p94_acc_.h5", "config5p1_17_4HL_6epochs_100p00_acc_.h5", "config5p2_17_4HL_7epochs_98p38_acc_.h5", "config5p3_17_4HL_6epochs_100p00_acc_.h5", "config5p4_17_4HL_13epochs_70p49_acc_.h5"]

if (devFlag):
	_, devData,_ = classify_general.getData(testDir, testConfig, trainSize, trainFlag, devFlag, testFlag)
	x_dev, y_dev_oh = devData
	
	for index, modelName in enumerate(modelList):
		modelPath = modelDir + "/" + configList[count] + "/" + modelList[count]
		print("Loading model \n%s\n" %(modelPath))
		model = load_model(modelPath)
		print("Finished loading model \n%s\n" %(modelPath))

		## Evaluate the performance of model on testData
		model_score = model.evaluate(x_dev, y_dev_oh, batch_size=2048)
		print("\nmodel= %s score on %s is: %s\n" %(configList[count], testConfig, model_score[1]))
		
		## Convert from one-hot to numerical prediction
		y_pred = np.argmax(model.predict(x_dev, batch_size=2048), axis=1)
		
		## vstack the actual and predicted output and take transpose
		output_predict = np.vstack((np.argmax(y_dev_oh, axis=1), y_pred)).T
		
		## Save it to csv file for future analysis
		outputFile = modelDir + "/"  + testConfig + "/data_" + testConfig + "_modelOf_" + str(configList[count])  + "_dev.csv" 
		np.savetxt(outputFile, output_predict, fmt="%5.0f", delimiter=",")
		
		##Error Analysis
		errorAnalysis(outputFile)
		count = count + 1
		## Only doing 4 config's at a time, 4p2 error during load_model
		if (count %4 == 0):
			break

elif (testFlag):
	testDataPath = testDir + "/" + testConfig + "/"
	_, _, testData = classify_general.getData(testDir, testConfig, trainSize, trainFlag, devFlag, testFlag)
	x_test, y_test_oh = testData
	
	for index, modelName in enumerate(modelList):
		modelPath = modelDir + "/" + configList[count] + "/" + modelList[count]
		model = load_model(modelPath)

		## Evaluate the performance of model on testData
		model_score = model.evaluate(x_test, y_test_oh, batch_size=2048)
		print("\nmodel= %s score on %s is: %s\n" %(configList[count], testConfig, model_score[1]))
		
		## Convert from one-hot to numerical prediction
		y_pred = np.argmax(model.predict(x_test, batch_size=2048), axis=1)
		
		## vstack the actual and predicted output and take transpose
		output_predict = np.vstack((np.argmax(y_test_oh, axis=1), y_pred)).T
		
		## Save it to csv file for future analysis
		outputFile = modelDir + "/"  + testConfig + "/data_" + testConfig + "_modelOf_" + str(configList[count])  + "_test.csv" 
		np.savetxt(outputFile, output_predict, fmt="%5.0f", delimiter=",")
		
		##Error Analysis
		errorAnalysis(outputFile)
		count + count + 1
		if (count %4 == 0):
			break

print("#########--------------###########\n")
