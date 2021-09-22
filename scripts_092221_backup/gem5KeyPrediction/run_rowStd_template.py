import sys
sys.path.insert(0, '/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/')
import classify_general
import newLoadData
import time
import random
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from optparse import OptionParser

parser = OptionParser()
parser.add_option('--trainSize',
									action = 'store', type='int', dest='trainSize', default = 28000)
parser.add_option('--resultDir',
									action = 'store', type='string', dest='resultDir', default = '/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/')
parser.add_option('--modelName',
									action = 'store', type='string', dest='modelName', default = 'gem5Model')
parser.add_option('--configName',
									action = 'store', type='string', dest='configName', default = 'config3p1')
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
configName = options.configName
trainFlag = options.trainFlag
devFlag = options.devFlag
testFlag = options.testFlag
numPowerTraces = options.numPowerTraces

dataDir = "/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/"
data=newLoadData.Data()
x_train, y_train = data.getData(dataDir, configName, trainSize, "Train")
x_train, y_train = data.shuffleData(x_train, y_train)
y_train_oh = data.oneHotY(y_train)

## Save images of first 5 traces before and after norm
plt_x = np.linspace(0,numPowerTraces-1, num=numPowerTraces)
for index in range(5):
	plt.plot(plt_x, x_train.iloc[index], 'b')
	figName = resultDir + "/" + configName + "/images_debug/" + modelName + "_train_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_preStd_key" + str(y_train.iloc[index].values[0]) + ".png"
	plt.savefig(figName)
	plt.close()

## RowStd
x_train = data.stdDataRowWise(x_train.to_numpy())
for index in range(5):
	plt.plot(plt_x, x_train[index], 'k')
	figName = resultDir + "/" + configName + "/images_debug/" + modelName + "_train_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_postStd_key" + str(y_train.iloc[index].values[0]) + ".png"
	plt.savefig(figName)
	plt.close()

gc.collect()
print("\nGarbage collected after train\n")

x_dev, y_dev = data.getData(dataDir, configName, 1000, "Dev")
x_dev, y_dev = data.shuffleData(x_dev, y_dev)
y_dev_oh = data.oneHotY(y_dev)

for index in range(5):
	plt.plot(plt_x, x_dev.iloc[index], 'r')
	figName = resultDir + "/" + configName + "/images_debug/" + modelName + "_dev_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_preStd_key" + str(y_dev.iloc[index].values[0]) + ".png"
	plt.savefig(figName)
	plt.close()

x_dev = data.stdDataRowWise(x_dev.to_numpy())
for index in range(5):
	plt.plot(plt_x, x_dev[index], 'g')
	figName = resultDir + "/" + configName + "/images_debug/" + modelName + "_dev_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_postStd_key" + str(y_dev.iloc[index].values[0]) + ".png"
	plt.savefig(figName)
	plt.close()

gc.collect()
print("\nGarbage collected after dev\n")

x_test, y_test = data.getData(dataDir, configName, 100, "Test")
x_test, y_test = data.shuffleData(x_test, y_test)
y_test_oh = data.oneHotY(y_test)
x_test = data.stdDataRowWise(x_test.to_numpy())
gc.collect()
print("\nGarbage collected after test\n")

##This is from the classify_general
#MtrainData, devData, testData = classify_general.getData(dataDir, configName, trainSize, trainFlag, devFlag, testFlag)
#M
#Mx_train, y_train_oh = trainData
#Mx_dev, y_dev_oh = devData
#Mx_test, y_test_oh = testData

## Instantiate the model and test, dev and training sets
##resultDir = "/extra/manojgopale/AES_data/config3p1_15ktraining/result_new"
##modelName = "m_newscript"
np.random.seed()

#numAllHiddenLayers = [3,4,5,6,7,8,9,10]
numAllHiddenLayers = [1,2,3,4]
hiddenLayerDict = {"num": [1,2,3,4,5,6,7,8,9,10], "factor": [10, 100, 1000]}
allAct = ['relu', 'tanh', 'elu']
allDrop = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
batchNormBin = [0, 1] ##Disabled batch norm to check if the runs go through
## Lower batizes did not yield good results, starting from 2^10
#batchSizePowers = [5,6,7,8,9,10,11,12,13,14,15,16]
#batchSizePowers = [10,11,12,13,14,15,16] ##Till run170
batchSizePowers = [10,11,12,13]
allOpt = ['Adam', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']

## number of samples is not required, since default is none, 
## and also will return an array if samples is provided, which wont work while indexing
numHiddenLayers = numAllHiddenLayers[np.random.random_integers(0, len(numAllHiddenLayers)-1)]
actList = [allAct[i] for i in np.random.random_integers(0, len(allAct)-1, numHiddenLayers).tolist()]
dropList = [allDrop[i] for i in np.random.random_integers(0, len(allDrop)-1, numHiddenLayers).tolist()]
batchNorm = [batchNormBin[i] for i in np.random.random_integers(0, len(batchNormBin)-1, numHiddenLayers).tolist()]
batchSize = np.power(2, batchSizePowers[np.random.random_integers(0, len(batchSizePowers)-1)])
learingRate = np.float_power(10, np.random.random_integers(-3,0))
epsilonValue = np.float_power(10, np.random.random_integers(-7, 0))
#MoptStr = random.sample(allOpt, 1)
optStr = "Adam" ## Hardcoding Adam for initial runs
learningRate = np.float_power(10,-3) ## Hardcoding epsilon and lr to default in initial runs
epsilonValue = np.float_power(10, -7)

#hiddenLayer = np.array([hiddenLayerDict["num"][i] for i in np.random.random_integers(0, len(hiddenLayerDict["num"])-1, numHiddenLayers).tolist()]) * np.array([hiddenLayerDict["factor"][i] for i in np.random.random_integers(0, len(hiddenLayerDict["factor"])-1, numHiddenLayers).tolist()])
## Get random integers between 100, 1500 , those seems to be giving better results
#MhiddenLayer = np.random.randint(100, 10000, numHiddenLayers)
hiddenLayer = [np.power(2,i) for i in np.random.random_integers(5,9, size=numHiddenLayers)]## runs 1-10 with powers of 2

runLogsPath = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/log/" + configName + "/allRuns.csv"
with open(runLogsPath, 'a') as f:
	## modelName must be unique like run_<someNum>
	f.write("\n%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n" %(modelName, numHiddenLayers, hiddenLayer, actList, dropList, batchNorm, batchSize, trainSize, learningRate, epsilonValue, optStr))

t0_time = time.time()
## This is for x_train is not pandas df
#Mclassifier = classify_general.Classifier(resultDir, modelName, x_train, y_train_oh, x_dev, y_dev_oh, x_test, y_test_oh, hiddenLayer, actList, dropList, batchNorm, numPowerTraces, configName)
print("type of, x_train=%s, y_train_oh=%s\n" %(type(x_train), type(y_train_oh)))
classifier = classify_general.Classifier(resultDir, modelName, x_train, y_train_oh, x_dev, y_dev_oh, x_test, y_test_oh, hiddenLayer, actList, dropList, batchNorm, numPowerTraces, configName, learningRate, epsilonValue, optStr)
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
