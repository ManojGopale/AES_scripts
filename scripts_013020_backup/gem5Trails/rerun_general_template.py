import sys
sys.path.insert(0, '/xdisk/rlysecky/manojgopale/extra/gem5Trails/scr/')
import classify_general
import time
import numpy as np

from optparse import OptionParser

parser = OptionParser()
parser.add_option('--trainSize',
									action = 'store', type='int', dest='trainSize', default = 15000)
parser.add_option('--resultDir',
									action = 'store', type='string', dest='resultDir', default = '/xdisk/rlysecky/manojgopale/extra/gem5Trails/result/')
parser.add_option('--modelName',
									action = 'store', type='string', dest='modelName', default = 'gem5Model')
parser.add_option('--trainFlag',
									action = 'store', type='int', dest='trainFlag', default = 1)
parser.add_option('--devFlag',
									action = 'store', type='int', dest='devFlag', default = 1)
parser.add_option('--testFlag',
									action = 'store', type='int', dest='testFlag', default = 0)
parser.add_option('--numPowerTraces',
									action = 'store', type='int', dest='numPowerTraces', default = 1361)
parser.add_option('--config',
									action = 'store', type='string', dest='config', default = 'config3p1')

(options, args) = parser.parse_args()

########
trainSize = options.trainSize
resultDir = options.resultDir
modelName = options.modelName
trainFlag = options.trainFlag
devFlag = options.devFlag
testFlag = options.testFlag
numPowerTraces = options.numPowerTraces
config = options.config

dataDir = "/xdisk/rlysecky/manojgopale/extra/gem5Trails/data/"
trainData, devData, testData = classify_general.getData(dataDir, config, trainSize, trainFlag, devFlag, testFlag)

x_train, y_train_oh = trainData
x_dev, y_dev_oh = devData
x_test, y_test_oh = testData

## Instantiate the model and test, dev and training sets
##resultDir = "/extra/manojgopale/AES_data/config3p1_15ktraining/result_new"
##modelName = "m_newscript"
np.random.seed()
allAct = ['relu', 'tanh', 'elu']
allDrop = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
batchSizePowers = [10,11,12,13]
batchNormBin = [0, 1] ##Disabled batch norm to check if the runs go through

numHiddenLayers =  4
#actList = [allAct[i] for i in np.random.random_integers(0, len(allAct)-1, numHiddenLayers).tolist()]
#dropList = [allDrop[i] for i in np.random.random_integers(0, len(allDrop)-1, numHiddenLayers).tolist()]
#batchNorm = [batchNormBin[i] for i in np.random.random_integers(0, len(batchNormBin)-1, numHiddenLayers).tolist()]
#batchSize = np.power(2, batchSizePowers[np.random.random_integers(0, len(batchSizePowers)-1)])

actList = ['relu', 'elu', 'tanh', 'elu']
dropList = [0.4, 0.5, 0.5, 0]
batchNorm = [0, 1, 0, 0]
batchSize = 2048
hiddenLayer = [7951, 9927, 7256, 2594]

runLogsPath = "/xdisk/rlysecky/manojgopale/extra/gem5Trails/log/" + config + "/allRuns.csv"
with open(runLogsPath, 'a') as f:
	## modelName must be unique like run_<someNum>
	f.write("\n%s, %s, %s, %s, %s, %s, %s, %s\n" %(config, modelName, numHiddenLayers, hiddenLayer, actList, dropList, batchNorm, batchSize))

t0_time = time.time()
#classifier = classify_general.Classifier(resultDir, modelName, x_train, y_train_oh, x_dev, y_dev_oh, x_test, y_test_oh, hiddenLayer, actList, dropList, batchNorm)
## Taking 1361 power traces only

## For gem5 runs, resultDir is within its coressponding folders
resultDir = resultDir + "/" + config + "/"

classifier = classify_general.Classifier(resultDir, modelName, x_train, y_train_oh, x_dev, y_dev_oh, x_test, y_test_oh, hiddenLayer, actList, dropList, batchNorm, numPowerTraces)
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
classifier.keyAccuracy(config)
