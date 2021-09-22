import sys
sys.path.insert(0, '/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/')
import classify_general
import newLoadData
import time
import numpy as np
import gc
import pandas as pd
import matplotlib.pyplot as plt
import random
from itertools import combinations

from optparse import OptionParser

def get_comma_separated_args(option, opt, value, parser):
	print("Inside callback\n")
	setattr(parser.values, option.dest, value.split(','))

parser = OptionParser()
parser.add_option('--trainSize',
									action = 'store', type='int', dest='trainSize', default = 28000)
parser.add_option('--resultDir',
									action = 'store', type='string', dest='resultDir', default = '/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/')
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
parser.add_option('--typeOfStd',
									action = 'store', type='string', dest='typeOfStd', default = 'col')
parser.add_option('--combIndex',
									action = 'store', type='int', dest='combIndex', default = 0)
parser.add_option('--configList',
                  type='string',
									action='callback',
									callback=get_comma_separated_args,
									dest = "getConfigList")

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
typeOfStd = options.typeOfStd
combIndex = options.combIndex #Index of the combination, we want to use for trials
argConfigList = options.getConfigList ## List of comfigs that can be given to script

np.random.seed()

## Load meand and std dev 
meanPath = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/allConfigs/config3p1_3p2_3p3_3p4_4p1_4p2_4p3_4p4_5p1_5p2_5p3_5p4_mean.csv"
stdPath = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/allConfigs/config3p1_3p2_3p3_3p4_4p1_4p2_4p3_4p4_5p1_5p2_5p3_5p4_std.csv"
## Converting them to numpy array for standardisation
mean_pool = pd.read_csv(meanPath, header=None).to_numpy()
std_pool = pd.read_csv(stdPath, header=None).to_numpy()
print("Loaded mean and std files from\n%s\n%s\n" %(meanPath, stdPath))
## Reshaping so that it matches the standardization function and not error out
mean_pool = mean_pool.reshape(mean_pool.shape[0], )
std_pool = std_pool.reshape(std_pool.shape[0], )

dataDir = "/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/"
data=newLoadData.Data()

#configList = ["config3p1", "config3p2", "config3p3", "config3p4", "config4p1", "config4p2", "config4p3", "config4p4", "config5p1", "config5p2", "config5p3", "config5p4"]
#configList = ["config3p3", "config4p1", "config5p4"]

## Instead of random_integers it should be a non-replaceable sample
#MtrainConfigs = [configList[i] for i in np.random.random_integers(0, len(configList)-1, numConfig).tolist()]
##Create a name by appending the config numbers, uselful for saving files
if(argConfigList[0] == "random"):
	configList = ["config3p1", "config3p2", "config3p3", "config3p4", "config4p1", "config4p2", "config4p3", "config4p4", "config5p1", "config5p2", "config5p3", "config5p4"]
	# Get random sample
	trainConfigs = random.sample(configList, numConfig)
elif (argConfigList[0] == "all"):
	trainConfigs = ["config3p1", "config3p2", "config3p3", "config3p4", "config4p1", "config4p2", "config4p3", "config4p4", "config5p1", "config5p2", "config5p3", "config5p4"]
elif (argConfigList[0] == "combination"):
	## The comb will always generate in the same order. Hence we can be sure to use indexing to get the triplets
	trainConfigs = list(combinations(combList, 3))[combIndex]
else:
	trainConfigs = argConfigList

	
configNames = "config_" + "_".join([i[-3:] for i in trainConfigs])
print("configName= %s\n" %(configNames))
x_train = pd.DataFrame()
y_train = pd.DataFrame()
x_dev = pd.DataFrame()
y_dev = pd.DataFrame()

for index, config in enumerate(trainConfigs):
	print("Loading data for %s\n" %(config))
	## Load data from config's
	## Append them
	## Shuffle and standardize them

	x_train_inter, y_train_inter = data.getData(dataDir, config, trainSize, "Train")
	x_train_inter, y_train_inter= data.shuffleData(x_train_inter, y_train_inter)
	x_train = pd.concat([x_train, x_train_inter], ignore_index=True)
	y_train = pd.concat([y_train, y_train_inter], ignore_index=True)

	x_dev_inter, y_dev_inter = data.getData(dataDir, config, 10, "Dev")
	x_dev_inter, y_dev_inter= data.shuffleData(x_dev_inter, y_dev_inter)
	x_dev = pd.concat([x_dev, x_dev_inter], ignore_index=True)
	y_dev = pd.concat([y_dev, y_dev_inter], ignore_index=True)
## Shuffle and standardize
print("After concatenation shapes are x_train= %s, y_train= %s, x_dev=%s, y_dev= %s\n" %(x_train.shape, y_train.shape, x_dev.shape, y_dev.shape))

x_train, y_train = data.shuffleData(x_train, y_train)

plt_x = np.linspace(0,numPowerTraces-1, num=numPowerTraces)
for index in range(5):
	plt.plot(plt_x, x_train.iloc[index], 'b')
	figName = resultDir  + "/dataEnsemble/images_debug/" + configNames + "_train_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_preStd_key" + str(y_train.iloc[index].values[0]) + ".png"
	plt.savefig(figName)
	plt.close()

## Standardize
if (typeOfStd == "col"):
	x_train = data.stdData(x_train.to_numpy(), mean_pool, std_pool)
	x_train = np.where(x_train>10, 10, x_train)
	x_train = np.where(x_train<-10, -10, x_train)
elif(typeOfStd == "row"):
	x_train = data.stdDataRowWise(x_train.to_numpy())

for index in range(5):
	plt.plot(plt_x, x_train[index], 'k')
	figName = resultDir + "/dataEnsemble/images_debug/" + configNames + "_train_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_postStd_key" + str(y_train.iloc[index].values[0]) + ".png"

	plt.savefig(figName)
	plt.close()

gc.collect()
print("\nGarbage collected after train\n")


x_dev, y_dev = data.shuffleData(x_dev, y_dev)

for index in range(5):
	plt.plot(plt_x, x_dev.iloc[index], 'b')
	figName = resultDir + "/dataEnsemble/images_debug/" + configNames + "_dev_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_preStd_key" + str(y_dev.iloc[index].values[0]) + ".png"
	plt.savefig(figName)
	plt.close()

## Standardize
if (typeOfStd == "col"):
	x_dev = data.stdData(x_dev.to_numpy(), mean_pool, std_pool)
	x_dev = np.where(x_dev>10, 10, x_dev)
	x_dev = np.where(x_dev<-10, -10, x_dev)
elif(typeOfStd == "row"):
	x_dev = data.stdDataRowWise(x_dev.to_numpy())

for index in range(5):
	plt.plot(plt_x, x_dev[index], 'k')
	figName = resultDir + "/dataEnsemble/images_debug/" + configNames + "_dev_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_postStd_key" + str(y_dev.iloc[index].values[0]) + ".png"
	plt.savefig(figName)
	plt.close()

gc.collect()
print("\nGarbage collected after dev\n")

##One hot the y_train and y_dev
print("Started one hot\n")
y_train_oh = data.oneHotY(y_train)
y_dev_oh = data.oneHotY(y_dev)
print("Done one hot\n")


print("After standardizing shapes are x_train= %s, y_train_oh= %s, x_dev=%s, y_dev_oh= %s\n" %(x_train.shape, y_train_oh.shape, x_dev.shape, y_dev_oh.shape))

## NOTE: DElete after one runs goes through
###M	trainData, devData, testData = classify_general_dataEnsemble.getData(dataDir, config, trainSize, trainFlag, devFlag, testFlag)
###M	x_train_inter, y_train_inter = trainData
###M	x_dev_inter, y_dev_inter = devData
###M	x_test_inter, y_test_inter = testData
###M	if(index == 0 ):
###M		x_train = x_train_inter
###M		x_dev   = x_dev_inter
###M		x_test  = x_test_inter
###M		y_train = y_train_inter
###M		y_dev   = y_dev_inter
###M		y_test  = y_test_inter
###M	else:
###M		x_train = np.concatenate((x_train_inter, x_train), 0)
###M		x_dev   = np.concatenate((x_dev_inter, x_dev), 0)
###M		x_test  = np.concatenate((x_test_inter, x_test), 0)
###M		y_train = np.concatenate((y_train, y_train_inter), 0)
###M		y_dev   = np.concatenate((y_dev, y_dev_inter), 0)
###M		y_test  = np.concatenate((y_test, y_test_inter), 0)
###M	print("Done loading data for %s\n" %(config))
###M
###Mx_train_inter = None
###Mx_dev_inter = None
###Mx_test_inter = None
###My_train_inter = None
###My_dev_inter = None
###My_test_inter = None
###M
###Mgc.collect()
###Mprint("Garbage collected\n")
###M	
###Mprint("After concatenation shapes are x_train= %s, y_train= %s, x_dev=%s, y_dev= %s\n" %(x_train.shape, y_train.shape, x_dev.shape, y_dev.shape))
###Mprint("Last x_train\n")
###Mprint(*x_train[-1])
###Mprint("Last x_dev\n")
###Mprint(*x_dev[-1])
###M##We have introduced a new function stdData, that standardises the data, one-hot's the y data
###M## Standardising it sepearatley and then concatenating did not give good results, lets see how it works when we add them together
###MstdData = classify_general_dataEnsemble.StandardizeData(x_train, y_train, x_dev, y_dev, x_test, y_test)
###M
###M## Shuffle train and dev data
###Mprint("Started shuffling\n")
###M(x_train, y_train) = stdData.shuffleData(x_train, y_train)
###M(x_dev, y_dev) = stdData.shuffleData(x_dev, y_dev)
###M(x_test, y_test) = stdData.shuffleData(x_test, y_test)
###Mprint("Done shuffling\n")
###M
###M##One hot the y_train and y_dev
###Mprint("Started one hot\n")
###My_train_oh = stdData.oneHotY(y_train)
###My_dev_oh = stdData.oneHotY(y_dev)
###My_test_oh = stdData.oneHotY(y_test)
###Mprint("Done one hot\n")
###M
###M##Deleting space occupied by y_*
###My_train = None
###My_dev = None
###My_test = None
###M
###Mgc.collect()
###Mprint("Garbage collected\n")
###M##Standardize the data by train data only
###Mprint("Started standardizing\n")
###Mx_train1 = stdData.stdByTrain(x_train)
###Mx_dev1 = stdData.stdByTrain(x_dev)
###Mx_test1 = stdData.stdByTrain(x_test)
###Mprint("Done standardizing\n")
###M
###Mprint("After standardizing shapes are x_train1= %s, y_train_oh= %s, x_dev1=%s, y_dev_oh= %s\n" %(x_train1.shape, y_train_oh.shape, x_dev1.shape, y_dev_oh.shape))
###Mprint("Last x_train1\n")
###Mprint(*x_train1[-1])
###Mprint("Last x_dev1\n")
###Mprint(*x_dev1[-1])


## Instantiate the model and test, dev and training sets
##resultDir = "/extra/manojgopale/AES_data/config3p1_15ktraining/result_new"
##modelName = "m_newscript"

#numAllHiddenLayers = [3,4,5,6,7,8,9,10]
numAllHiddenLayers = [1,2,3,4]
hiddenLayerDict = {"num": [1,2,3,4,5,6,7,8,9,10], "factor": [10, 100, 1000]}
allAct = ['relu', 'tanh', 'elu']
allDrop = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
batchNormBin = [0, 1] ##Disabled batch norm to check if the runs go through
## Lower batizes did not yield good results, starting from 2^10
#batchSizePowers = [5,6,7,8,9,10,11,12,13,14,15,16]
#batchSizePowers = [10,11,12,13,14,15,16] ##Till run170
batchSizePowers = [5,6,7,8,9]
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
optStr = random.sample(allOpt, 1)
optStr = "Adam" ## Hardcoding Adam for initial runs
learningRate = np.float_power(10,-3) ## Hardcoding epsilon and lr to default in initial runs
epsilonValue = np.float_power(10, -7)

#hiddenLayer = np.array([hiddenLayerDict["num"][i] for i in np.random.random_integers(0, len(hiddenLayerDict["num"])-1, numHiddenLayers).tolist()]) * np.array([hiddenLayerDict["factor"][i] for i in np.random.random_integers(0, len(hiddenLayerDict["factor"])-1, numHiddenLayers).tolist()])
## Get random integers between 100, 1500 , those seems to be giving better results
#hiddenLayer = np.random.randint(100, 10000, numHiddenLayers) 
hiddenLayer = [np.power(2,i) for i in np.random.random_integers(5,9, size=numHiddenLayers)]## runs 30 onwards with powers of 2

runLogsPath = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/log/dataEnsemble/allRuns.csv"
with open(runLogsPath, 'a') as f:
	## modelName must be unique like run_<someNum>
	f.write("\n%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n" %(modelName, numHiddenLayers, hiddenLayer, actList, dropList, batchNorm, batchSize, trainSize, learningRate, epsilonValue, optStr, typeOfStd))

t0_time = time.time()
classifier = classify_general.Classifier(resultDir, modelName, x_train, y_train_oh, x_dev, y_dev_oh, x_dev, y_dev_oh, hiddenLayer, actList, dropList, batchNorm, numPowerTraces, "dataEnsemble", learningRate, epsilonValue, optStr)
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
classifier.keyAccuracy(trainSize)
