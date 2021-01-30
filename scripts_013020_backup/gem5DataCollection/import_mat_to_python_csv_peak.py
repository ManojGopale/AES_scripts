import scipy.io as si
import numpy as np
import argparse
import pickle
import gzip
import time
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-c','--config',
									action = 'store', type='string', dest='config', default = 'config3p1')
parser.add_option('-d','--dataDir',
									action = 'store', type='string', dest='dataDir', default = '/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/')
parser.add_option('--partLen',
									action = 'store', type='int', dest='partLen', default = 64)

(options, args) = parser.parse_args()

startTime = time.time()
#parser = argparse.ArgumentParser()
#parser.add_argument("key", help="input the key value to add to the training and testing data", type=int)
#args = parser.parse_args()

## keySize must be divisible by partLen exactly
keySize = 256
count = 0

## File paths
## This is where the *.mat files are saved
parentDir = options.dataDir
configName = options.config
partLen = options.partLen

logFile = parentDir + '/csvResult/' + configName + '/import_mat_to_csv.log'

with open(logFile, 'w') as f:
	for key in range(keySize):
		keyStart = time.time()
		keyStr = parentDir  + '/matResult/' + configName + '/value' + str(key) + '.mat'
		print ("\nKey str = %s" %(keyStr))
		f.write ("\nKey str = %s" %(keyStr))
		
		keyData = si.loadmat(keyStr)
		#print ("key %d, keyData shape= %s" %(key, keyData['power'].shape))
		
		keyDataArray = keyData['power'].reshape(keyData['power'].shape)[:,0:1500]
		#print ("keyDataArray shape = {0}" .format(keyDataArray.shape))
		
		## Appending ouput value to the array as column
		fullKeyData = np.concatenate((keyDataArray, np.zeros([keyData['power'].shape[0], 1])), 1)
		#print ("fullKeyData shape = {0}" .format(fullKeyData.shape))
		
		## substituting the last coulmn with the key value
		fullKeyData[:, -1] = key
		#print ("fullKeyData shape = {0}" .format(fullKeyData.shape))
		#print ("Class value = %s" %(fullKeyData[:,-1]))
		
		## first 20000 as training and last 10000 as testing
		##trainKey = fullKeyData[0:20000, ]
		##testKey = fullKeyData[-10000:]

		## Testing with 28000 training, 1000 validation and 1000 testing for each key 
		trainKeyData = fullKeyData[0:29000, ]
		trainKey = trainKeyData[0:28000, ]
		devKey = trainKeyData[-1000:]
		testKey = fullKeyData[-1000:]
		#print ("Shapes of train and test are {0}, {1}" .format(trainKey.shape, testKey.shape))
		#print ("Shapes of train and test are {0}, {1}" .format(trainKey.shape, fullKeyData[-1000:-1,].shape))
		
		if key == 0 or key % partLen == 0:
			fullTrain = trainKey[:]
			fullDev = devKey[:] 
			fullTest = testKey[:]
			#print ("Shapes of fullTrain and fullTest are {0}, {1}" .format(fullTrain.shape, fullTest.shape))
		elif key % partLen == partLen-1:
			fullTrain = np.concatenate((fullTrain, trainKey), 0)
			fullDev = np.concatenate((fullDev, devKey), 0)
			fullTest = np.concatenate((fullTest, testKey), 0)
			## This loop is to dump the data when the key is one less than the divisor of partLen
			csvStrTrain = parentDir + '/csvResult/' + configName + '/aesData_config9_Train_' + str(count) + '.csv'
			print("csvStrTrain Dir = %s" %(csvStrTrain))
			np.savetxt(csvStrTrain, fullTrain, fmt="%10.5f", delimiter=",")

			csvStrDev = parentDir + '/csvResult/' + configName + '/aesData_config9_Dev_' + str(count) + '.csv'
			np.savetxt(csvStrDev, fullDev, fmt="%10.5f", delimiter=",")

			csvStrTest = parentDir + '/csvResult/' + configName + '/aesData_config9_Test_' + str(count) + '.csv'
			np.savetxt(csvStrTest, fullTest, fmt="%10.5f", delimiter=",")

			count = count + 1
			dumpTime = time.time()
			print ("DumpTime = %f" %(dumpTime-keyStop))
			f.write ("DumpTime = %f" %(dumpTime-keyStop))
			print ("\n Total Execution Time = %f sec" %(dumpTime - startTime))
			f.write ("\n Total Execution Time = %f sec\n" %(dumpTime - startTime))
		else:
			## Appending training and testing from current key to previous runs
			fullTrain = np.concatenate((fullTrain, trainKey), 0)
			fullDev = np.concatenate((fullDev, devKey), 0)
			fullTest = np.concatenate((fullTest, testKey), 0)
			#print ("Shapes of fullTrain, fullDev  and fullTest are {0}, {1}, {2}" .format(fullTrain.shape, fullDev.shape, fullTest.shape))
	
		keyStop = time.time()
		print ("Key= %d, execution time = %f sec" %(key, keyStop-keyStart))
		f.write ("Key= %d, execution time = %f sec" %(key, keyStop-keyStart))
	
	print ("\n Total Time to load fullTrain and fullTest = %f sec" %(keyStop - startTime))
	f.write ("\n Total Time to load fullTrain and fullTest = %f sec" %(keyStop - startTime))
	## Since fullTrain.shape is a tuple we need {} to reference them inside
	print ("Shapes of fullTrain, fullDev  and fullTest are {0}, {1}, {2}" .format(fullTrain.shape, fullDev.shape, fullTest.shape))
	f.write ("Shapes of fullTrain, fullDev and fullTest are {0}, {1}, {2}" .format(fullTrain.shape, fullDev.shape, fullTest.shape))
	
	### Create a pickle of the training and testing data for further use
	#with gzip.open('aesCpuData.pkl.gz', 'wb') as f:
	#	cPickle.dump([fullTrain, fullTest], f)
	#
	#dumpTime = time.time()
	#print ("DumpTime = %f" %(dumpTime-keyStop))
	#f.write ("DumpTime = %f" %(dumpTime-keyStop))
	#print ("\n Total Execution Time = %f sec" %(dumpTime - startTime))
	#f.write ("\n Total Execution Time = %f sec\n" %(dumpTime - startTime))
	
	## Opening the pkl.gz from previous runs to load the train and test data for append
	#with gzip.open('./pythonDir/aesCpuData.pkl.gz', 'rb') as f:
	#	trainLoad, testLoad = cPickle.load(f)
	#
	#loadTime = time.time()
	#print ("LoadTime = %f sec" %(loadTime-dumpTime))
	#f.write ("LoadTime = %f sec" %(loadTime-dumpTime))

