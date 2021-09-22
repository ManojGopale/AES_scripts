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
parser.add_option('--devSize',
									action = 'store', type='int', dest='devSize', default = 1000)
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
trainSize = options.trainSize
devSize = options.devSize
resultDir = options.resultDir
modelName = options.modelName
configName = options.configName
trainFlag = options.trainFlag
devFlag = options.devFlag
testFlag = options.testFlag
numPowerTraces = options.numPowerTraces
typeOfStd = options.typeOfStd
combIndex = options.combIndex #Index of the combination, we want to use for trials
argConfigList = options.getConfigList ## List of comfigs that can be given to script

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

if(argConfigList[0] == "random"):
	configList = ["config3p1", "config3p2", "config3p3", "config3p4", "config4p1", "config4p2", "config4p3", "config4p4", "config5p1", "config5p2", "config5p3", "config5p4"]
	# Get random sample
	trainConfigs = random.sample(configList, numConfig)
elif (argConfigList[0] == "all"):
	trainConfigs = ["config3p1", "config3p2", "config3p3", "config3p4", "config4p1", "config4p2", "config4p3", "config4p4", "config5p1", "config5p2", "config5p3", "config5p4"]
elif (argConfigList[0] == "combination"):
	## The comb will always generate in the same order. Hence we can be sure to use indexing to get the triplets
	combList = ["config3p1", "config3p4", "config4p1", "config4p4", "config5p1", "config5p4"]
	trainConfigs = list(combinations(combList, 3))[combIndex]
else:
	trainConfigs = argConfigList

np.random.seed()

configNames = "config_" + "_".join([i[-3:] for i in trainConfigs])
print("configNames = %s\n" %(configNames))

#numAllHiddenLayers = [3,4,5,6,7,8,9,10]
numAllHiddenLayers = [3,4]
hiddenLayerDict = {"num": [1,2,3,4,5,6,7,8,9,10], "factor": [10, 100, 1000]}
allAct = ['relu', 'tanh', 'elu']
allDrop = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
batchNormBin = [0, 1] ##Disabled batch norm to check if the runs go through
## Lower batizes did not yield good results, starting from 2^10
#batchSizePowers = [5,6,7,8,9,10,11,12,13,14,15,16]
#batchSizePowers = [10,11,12,13,14,15,16] ##Till run170
batchSizePowers = [4,5,6,7]

## number of samples is not required, since default is none, 
## and also will return an array if samples is provided, which wont work while indexing
numHiddenLayers = numAllHiddenLayers[np.random.random_integers(0, len(numAllHiddenLayers)-1)]
hiddenLayer = [np.power(2,i) for i in np.random.random_integers(4,8, size=numHiddenLayers)]## 
actList = [allAct[i] for i in np.random.random_integers(0, len(allAct)-1, numHiddenLayers).tolist()]
dropList = [allDrop[i] for i in np.random.random_integers(0, len(allDrop)-1, numHiddenLayers).tolist()]
batchNorm = [batchNormBin[i] for i in np.random.random_integers(0, len(batchNormBin)-1, numHiddenLayers).tolist()]
batchSize = np.power(2, batchSizePowers[np.random.random_integers(0, len(batchSizePowers)-1)])
#MbatchSize = [np.power(2,np.random.random_integers(4,7, size=1))][0]##This is an array and we need to index to get the element. 

#hiddenLayer = np.array([hiddenLayerDict["num"][i] for i in np.random.random_integers(0, len(hiddenLayerDict["num"])-1, numHiddenLayers).tolist()]) * np.array([hiddenLayerDict["factor"][i] for i in np.random.random_integers(0, len(hiddenLayerDict["factor"])-1, numHiddenLayers).tolist()])
## Get random integers between 100, 1500 , those seems to be giving better results
#MhiddenLayer = np.random.randint(100, 2000, numHiddenLayers) ##runs 1-20

optStr = "Adam" ## Hardcoding Adam for initial runs
learningRate = np.float_power(10,-3) ## Hardcoding epsilon and lr to default in initial runs
epsilonValue = np.float_power(10, -7)

x_train = pd.DataFrame()
y_train = pd.DataFrame()
x_dev = pd.DataFrame()
y_dev = pd.DataFrame()

for index, config in enumerate(trainConfigs):
	print("Loading data for %s\n" %(config))
	x_train_inter, y_train_inter = data.getData(dataDir, config, trainSize, "Train")
	x_train_inter, y_train_inter= data.shuffleData(x_train_inter, y_train_inter)
	x_train = pd.concat([x_train, x_train_inter], ignore_index=True)
	y_train = pd.concat([y_train, y_train_inter], ignore_index=True)

	x_dev_inter, y_dev_inter = data.getData(dataDir, config, devSize, "Dev")
	x_dev_inter, y_dev_inter= data.shuffleData(x_dev_inter, y_dev_inter)
	x_dev = pd.concat([x_dev, x_dev_inter], ignore_index=True)
	y_dev = pd.concat([y_dev, y_dev_inter], ignore_index=True)

print("After concatenation shapes are x_train= %s, y_train= %s, x_dev=%s, y_dev= %s\n" %(x_train.shape, y_train.shape, x_dev.shape, y_dev.shape))

x_train, y_train = data.shuffleData(x_train, y_train)
#######-----------------------------------------------------########
## Adding noise to the signals
## (mean, stdDev, shape) -> arguments to the function
#Mnoise = np.random.normal(0,0.001, x_train.shape)
#Mx_train_w_noise = x_train + noise
#Mx_train = pd.concat([x_train, x_train_w_noise])
#My_train = pd.concat([y_train, y_train])
#M
#Mprint("After noise addition shapes are x_train= %s, y_train= %s, x_dev=%s, y_dev= %s\n" %(x_train.shape, y_train.shape, x_dev.shape, y_dev.shape))
#######-----------------------------------------------------########

plt_x = np.linspace(0,numPowerTraces-1, num=numPowerTraces)
for index in range(5):
	plt.plot(plt_x, x_train.iloc[index], 'b')
	figName = resultDir  + configName + "/images_debug/" + configNames + "_train_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_preStd_key" + str(y_train.iloc[index].values[0]) + ".png"
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
	figName = resultDir + configName + "/images_debug/" + configNames + "_train_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_postStd_key" + str(y_train.iloc[index].values[0]) + ".png"

	plt.savefig(figName)
	plt.close()

gc.collect()
print("\nGarbage collected after train\n")


x_dev, y_dev = data.shuffleData(x_dev, y_dev)

for index in range(5):
	plt.plot(plt_x, x_dev.iloc[index], 'b')
	figName = resultDir + configName + "/images_debug/" + configNames + "_dev_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_preStd_key" + str(y_dev.iloc[index].values[0]) + ".png"
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
	figName = resultDir + configName + "/images_debug/" + configNames + "_dev_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_postStd_key" + str(y_dev.iloc[index].values[0]) + ".png"
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

runLogsPath = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/log/" + configName + "/allRuns.csv"
with open(runLogsPath, 'a') as f:
	## modelName must be unique like run_<someNum>
	f.write("\n%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n" %(modelName, numHiddenLayers, hiddenLayer, actList, dropList, batchNorm, batchSize, trainSize, learningRate, epsilonValue, optStr, typeOfStd, configNames))

t0_time = time.time()
classifier = classify_general.Classifier(resultDir, modelName, x_train, y_train_oh, x_dev, y_dev_oh, x_dev, y_dev_oh, hiddenLayer, actList, dropList, batchNorm, numPowerTraces, configName, learningRate, epsilonValue, optStr)
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
classifier.keyAccuracy(devSize)

## get perplexity
classifier.getPerplexity(x_train, y_train_oh, "Train")
classifier.getPerplexity(x_dev, y_dev_oh, "Dev")
