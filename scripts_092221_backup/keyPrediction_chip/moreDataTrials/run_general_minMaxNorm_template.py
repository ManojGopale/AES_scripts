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

dataDir = "/xdisk/rlysecky/manojgopale/extra/chipWhisperer_data/trace_key_1500/"
moreDataDir = "/xdisk/rlysecky/manojgopale/extra/chipWhisperer_data/trace_key_1500_1/"
trainData, devData, testData = classify_general_minMaxNorm.getData(dataDir, moreDataDir, trainSize, trainFlag, devFlag, testFlag)

x_train, y_train_oh = trainData
x_dev, y_dev_oh = devData
x_test, y_test_oh = testData

## Instantiate the model and test, dev and training sets
##resultDir = "/extra/manojgopale/AES_data/config3p1_15ktraining/result_new"
##modelName = "m_newscript"
np.random.seed()

numAllHiddenLayers = [3,4,5,6,7,8,9,10]
hiddenLayerDict = {"num": [1,2,3,4,5,6,7,8,9,10], "factor": [10, 100, 1000]}
allAct = ['relu', 'tanh', 'elu']
allDrop = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
batchNormBin = [0, 1] ##Disabled batch norm to check if the runs go through
## Lower batizes did not yield good results, starting from 2^10
batchSizePowers = [10,11,12,13,14,15,16]

## number of samples is not required, since default is none, 
## and also will return an array if samples is provided, which wont work while indexing
numHiddenLayers = numAllHiddenLayers[np.random.random_integers(0, len(numAllHiddenLayers)-1)]
actList = [allAct[i] for i in np.random.random_integers(0, len(allAct)-1, numHiddenLayers).tolist()]
dropList = [allDrop[i] for i in np.random.random_integers(0, len(allDrop)-1, numHiddenLayers).tolist()]
batchNorm = [batchNormBin[i] for i in np.random.random_integers(0, len(batchNormBin)-1, numHiddenLayers).tolist()]
batchSize = np.power(2, batchSizePowers[np.random.random_integers(0, len(batchSizePowers)-1)])

#hiddenLayer = np.array([hiddenLayerDict["num"][i] for i in np.random.random_integers(0, len(hiddenLayerDict["num"])-1, numHiddenLayers).tolist()]) * np.array([hiddenLayerDict["factor"][i] for i in np.random.random_integers(0, len(hiddenLayerDict["factor"])-1, numHiddenLayers).tolist()])
## Get random integers between 100, 1500 , those seems to be giving better results
##hiddenLayer = np.random.randint(100, 2500, numHiddenLayers) ##Till run ~50-60 ->130
hiddenLayer = np.random.randint(1000, 4000, numHiddenLayers) ##Runs 0->

runLogsPath = "/xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/scr/allRuns_minMaxNorm.csv"
with open(runLogsPath, 'a') as f:
	## modelName must be unique like run_<someNum>
	f.write("\n%s, %s, %s, %s, %s, %s, %s\n" %(modelName, numHiddenLayers, hiddenLayer, actList, dropList, batchNorm, batchSize))

t0_time = time.time()
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
