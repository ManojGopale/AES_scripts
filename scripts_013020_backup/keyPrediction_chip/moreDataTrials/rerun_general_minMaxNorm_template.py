import sys
sys.path.insert(0, '/xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/scr/')
import classify_general_minMaxNorm
import time
import numpy as np

from optparse import OptionParser

parser = OptionParser()
parser.add_option('--trainSize',
									action = 'store', type='int', dest='trainSize', default = 28000)
parser.add_option('--resultDir',
									action = 'store', type='string', dest='resultDir', default = '/xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/result/')
parser.add_option('--modelName',
									action = 'store', type='string', dest='modelName', default = 'chipWhispererModel')
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
trainSize = options.trainSize
resultDir = options.resultDir
modelName = options.modelName
trainFlag = options.trainFlag
devFlag = options.devFlag
testFlag = options.testFlag
numPowerTraces = options.numPowerTraces

dataDir = "/xdisk/bethard/mig2020/extra/manojgopale/AES_data/chipwhispererData/trace_key_1500/"
moreDataDir = "/xdisk/rlysecky/manojgopale/extra/chipWhisperer_data/trace_key_1500_1/"
trainData, devData, testData = classify_general_minMaxNorm.getData(dataDir, moreDataDir, trainSize, trainFlag, devFlag, testFlag)

x_train, y_train_oh = trainData
x_dev, y_dev_oh = devData
x_test, y_test_oh = testData

## Instantiate the model and test, dev and training sets
##resultDir = "/extra/manojgopale/AES_data/config3p1_15ktraining/result_new"
##modelName = "m_newscript"
np.random.seed()

numHiddenLayers = 4
actList = ['relu', 'relu', 'relu', 'relu']
dropList = [0.3, 0, 0.5, 0.1]
batchNorm = [1, 0, 0, 1]
batchSize = 32768
hiddenLayer = [7951, 9927, 7256, 2594]

runLogsPath = "/xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/scr/allRuns_minMaxNorm.csv"
with open(runLogsPath, 'a') as f:
	## modelName must be unique like run_<someNum>
	f.write("\n%s, %s, %s, %s, %s, %s, %s\n" %(modelName, numHiddenLayers, hiddenLayer, actList, dropList, batchNorm, batchSize))

t0_time = time.time()
#classifier = classify_general_minMaxNorm.Classifier(resultDir, modelName, x_train, y_train_oh, x_dev, y_dev_oh, x_test, y_test_oh, hiddenLayer, actList, dropList, batchNorm)
## Taking 1361 power traces only
classifier = classify_general_minMaxNorm.Classifier(resultDir, modelName, x_train, y_train_oh, x_dev, y_dev_oh, x_test, y_test_oh, hiddenLayer, actList, dropList, batchNorm, numPowerTraces)
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
