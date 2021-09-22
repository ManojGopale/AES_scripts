import sys
sys.path.insert(0, '/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/')
import classify_general_dataEnsemble_noNormFunction
import time
import numpy as np

from optparse import OptionParser

parser = OptionParser()
parser.add_option('--trainSize',
									action = 'store', type='int', dest='trainSize', default = 28000)
parser.add_option('--resultDir',
									action = 'store', type='string', dest='resultDir', default = '/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/dataEnsemble/result/')
parser.add_option('--modelName',
									action = 'store', type='string', dest='modelName', default = 'gem5Model')
parser.add_option('--numConfig',
									action = 'store', type='int', dest='numConfig', default = 1)
parser.add_option('--trainFlag',
									action = 'store', type='int', dest='trainFlag', default = 1)
parser.add_option('--devFlag',
									action = 'store', type='int', dest='devFlag', default = 1)
parser.add_option('--testFlag',
									action = 'store', type='int', dest='testFlag', default = 0)
parser.add_option('--numPowerTraces',
									action = 'store', type='int', dest='numPowerTraces', default = 1500)

(options, args) = parser.parse_args()

########
## trainSize controls the size of traces per key for each of the configs
trainSize = options.trainSize
resultDir = options.resultDir
modelName = options.modelName
numConfig = options.numConfig
trainFlag = options.trainFlag
devFlag = options.devFlag
testFlag = options.testFlag
numPowerTraces = options.numPowerTraces

np.random.seed()

dataDir = "/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/"

configList = ["config3p1", "config3p2", "config3p3", "config3p4", "config4p1", "config4p2", "config4p3", "config4p4", "config5p1", "config5p2", "config5p3", "config5p4"]

trainConfigs = [configList[i] for i in np.random.random_integers(0, len(configList)-1, numConfig).tolist()]
##Create a name by appending the config numbers, uselful for saving files
configNames = "config_" + "_".join([i[-3:] for i in trainConfigs])

numHiddenLayers = 4
actList = ['elu', 'relu', 'elu', 'tanh']
dropList = [0, 0.1, 0.1, 0.4]
batchNorm = [1, 0, 1, 1]
batchSize = 1024
hiddenLayer = [1186, 1675, 290, 1995]


for index, config in enumerate(trainConfigs):
	print("Loading data for %s\n" %(config))
	trainData, devData, testData = classify_general_dataEnsemble_noNormFunction.getData(dataDir, config, trainSize, trainFlag, devFlag, testFlag)
	x_train_inter, y_train_inter_oh = trainData
	x_dev_inter, y_dev_inter_oh = devData
	x_test_inter, y_test_inter_oh = testData
	if(index == 0 ):
		x_train = x_train_inter
		x_dev   = x_dev_inter
		x_test  = x_test_inter
		y_train_oh = y_train_inter_oh
		y_dev_oh   = y_dev_inter_oh
		y_test_oh  = y_test_inter_oh
	else:
		x_train = np.concatenate((x_train_inter, x_train), 0)
		x_dev   = np.concatenate((x_dev_inter, x_dev), 0)
		x_test  = np.concatenate((x_test_inter, x_test), 0)
		y_train_oh = np.concatenate((y_train_oh, y_train_inter_oh), 0)
		y_dev_oh   = np.concatenate((y_dev_oh, y_dev_inter_oh), 0)
		y_test_oh  = np.concatenate((y_test_oh, y_test_inter_oh), 0)
	print("Done loading data for %s\n" %(config))

x_train_inter = None
x_dev_inter = None
x_test_inter = None
y_train_inter_oh = None
y_dev_inter_oh = None
y_test_inter_oh = None

## Instantiate the model and test, dev and training sets
##resultDir = "/extra/manojgopale/AES_data/config3p1_15ktraining/result_new"
##modelName = "m_newscript"

runLogsPath = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/dataEnsemble/log/allRuns.csv"
with open(runLogsPath, 'a') as f:
	## modelName must be unique like run_<someNum>
	f.write("\n%s, %s, %s, %s, %s, %s, %s, %s, %s\n" %(trainConfigs, trainSize, modelName, numHiddenLayers, hiddenLayer, actList, dropList, batchNorm, batchSize))

t0_time = time.time()
classifier = classify_general_dataEnsemble_noNormFunction.Classifier(resultDir, modelName, x_train, y_train_oh, x_dev, y_dev_oh, x_test, y_test_oh, hiddenLayer, actList, dropList, batchNorm, numPowerTraces, configNames)
t1_time = time.time()
print("\nTime to load the dataset in python for training is %s seconds\n" %(t1_time-t0_time))

## Train the model
startTime = time.time()
classifier.train(batchSize)
endTime = time.time()
trainTime = endTime - startTime
print("\nTime to train with batchSize= %s is %s seconds\n" %(batchSize, trainTime))

## Evaluate
classifier.evaluate()

##Save the model
classifier.saveModel()

## run Key accuracy class
classifier.keyAccuracy()
