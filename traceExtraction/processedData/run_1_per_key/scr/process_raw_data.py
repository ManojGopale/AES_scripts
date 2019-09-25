import scipy.io as si
import numpy as np
import time

import matplotlib.pyplot as plt
import random
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-c','--config',
									action = 'store', type='string', dest='config', default = 'config3p1')
parser.add_option('-d','--dataDir',
									action = 'store', type='string', dest='dataDir', default = '/xdisk/manojgopale/AES/dataCollection/matResult/2019-06-16_20_20_default_cache/')
parser.add_option('-b', '--batch',
									action = 'store', type='int', dest='batchSize', default = 128)

(options, args) = parser.parse_args()

## This is where the *.mat files are saved
dataDir = options.dataDir
config = options.config

## Save file directory, directory where the csv files will be saved
saveDataDir = "/xdisk/manojgopale/AES/dataCollection/processedData/run_1_per_key/data/"

## LogDir, dir to dump log from the run
logDir = "/xdisk/manojgopale/AES/dataCollection/processedData/run_1_per_key/log/"
logFile = logDir + config + ".log"

keyLen = 256
batchSize = options.batchSize

## Append the last column with the binary output for the traces, @2000 its 1
## reshape so that we get dimensions for concatenation
outputArray = np.zeros(4000,).reshape(4000,1)
outputArray[1999] = 1

count = 0

with open(logFile, "w") as f:
	keyStart = time.time()
	for key in range(keyLen):
		loopStart = time.time()
		matStr = dataDir + "/" + "/value" + str(key) + ".mat"
		print("\nMatfile = %s" %(matStr))
		f.write("\nMatfile = %s" %(matStr))
		
		## Load data into keyData
		keyData = si.loadmat(matStr)["power"]
	
		## Get 3 random sample numbers for generating the train, dev and test data
		traceNum = random.sample(range(1, 500), 3)
		print("traceNum for train=%d, dev=%d, test=%d" %(traceNum[0], traceNum[1], traceNum[2]))
		f.write("traceNum for train=%d, dev=%d, test=%d" %(traceNum[0], traceNum[1], traceNum[2]))
	
		## Intilialize traces for new key
		trainTrace = []
		devTrace = []
		testTrace = []
	
		## Creating train, dev and test arrays
		trainTrace = keyData[traceNum[0]][np.array([range(i, i+1361) for i in range(4000)])]
		devTrace = keyData[traceNum[1]][np.array([range(i, i+1361) for i in range(4000)])]
		testTrace = keyData[traceNum[2]][np.array([range(i, i+1361) for i in range(4000)])]

		## Append the outputArray to the train, dev and test
		trainTrace = np.concatenate((trainTrace, outputArray), 1)
		devTrace = np.concatenate((devTrace, outputArray), 1)
		testTrace = np.concatenate((testTrace, outputArray), 1)
	
		print("trainTrace shape = %s" %(trainTrace.shape,))

		## appending to full traces
		if key == 0 or key % batchSize == 0:
			fullTrain = trainTrace[:]
			fullDev = devTrace[:]
			fullTest = testTrace[:]
		else:
			fullTrain = np.concatenate((fullTrain, trainTrace), 0)
			fullDev = np.concatenate((fullDev, devTrace), 0)
			fullTest = np.concatenate((fullTest, testTrace), 0)

		print("FullTrain shape = %s" %(fullTrain.shape,))

		if key % batchSize == batchSize - 1 or key == keyLen - 1:
			##Save the file when at batchSize-1 since we start from 0 indexing
			trainCsvFile = saveDataDir + config + "/train_" + str(count) + ".csv"
			devCsvFile = saveDataDir + config + "/dev_" + str(count) + ".csv"
			testCsvFile = saveDataDir + config + "/test_" + str(count) + ".csv"
	
			## Increment the count 
			count = count + 1
	
			##Save to csv files
			np.savetxt(trainCsvFile, fullTrain, fmt="%10.5f", delimiter=",")
			np.savetxt(devCsvFile, fullDev, fmt="%10.5f", delimiter=",")
			np.savetxt(testCsvFile, fullTest, fmt="%10.5f", delimiter=",")

			interStop= time.time()
			print("\nWritten files to \n%s\n%s\n%s" %(trainCsvFile, devCsvFile, testCsvFile))
			f.write("\nWritten files to \n%s\n%s\n%s" %(trainCsvFile, devCsvFile, testCsvFile))

			print("Total time since beginning till %s key is %f sec" %(key, interStop-keyStart))
			f.write("Total time since beginning till %s key is %f sec" %(key, interStop-keyStart))

		##
		loopEnd = time.time()
		print("Loop time = %s" %(loopEnd - loopStart))
		f.write("Loop time = %s" %(loopEnd - loopStart))
