import sys
sys.path.insert(0, '/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/')
import classify_general
import newLoadData
import time
import numpy as np
import gc
import pandas as pd
import matplotlib.pyplot as plt

from optparse import OptionParser

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

## Load meand and std dev 
## Added this to classify_general file
## NOTE: Commenting for 1361 runs, to be uncommented after the run is over
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
x_train, y_train = data.getData(dataDir, configName, trainSize, "Train")
x_train, y_train = data.shuffleData(x_train, y_train)
y_train_oh = data.oneHotY(y_train)
#Mmean_pool, std_pool = data.getStdParam(x_train) ## Comment after 1361 run is over
#Mmean_pool = mean_pool.to_numpy()
#Mmean_pool = mean_pool.reshape(mean_pool.shape[0], )
#Mstd_pool = std_pool.to_numpy()
#Mstd_pool = std_pool.reshape(mean_pool.shape[0], )

## Save images of first 5 traces before and after norm
plt_x = np.linspace(0,numPowerTraces-1, num=numPowerTraces)
for index in range(5):
	plt.plot(plt_x, x_train.iloc[index], 'b')
	figName = resultDir + "/" + configName + "/images_debug/" + modelName + "_train_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_preStd_key" + str(y_train.iloc[index].values[0]) + ".png"
	plt.savefig(figName)
	plt.close()

#Mx_train = data.stdData(x_train.to_numpy(), mean_pool, std_pool)
## Standardize
if (typeOfStd == "col"):
	x_train = data.stdData(x_train.to_numpy(), mean_pool, std_pool)
	x_train = np.where(x_train>10, 10, x_train)
	x_train = np.where(x_train<-10, -10, x_train)
elif(typeOfStd == "row"):
	x_train = data.stdDataRowWise(x_train.to_numpy())

## Commenting out the saving to file, because the error was not in data and it is huge is size
##Mtrain_std_path = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/" + configName + "/train_std_3p1_3p2.csv"
##Mnp.savetxt(train_std_path, x_train, delimiter=",")
for index in range(5):
	plt.plot(plt_x, x_train[index], 'k')
	figName = resultDir + "/" + configName + "/images_debug/" + modelName + "_train_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_postStd_key" + str(y_train.iloc[index].values[0]) + ".png"
	plt.savefig(figName)
	plt.close()

gc.collect()
print("\nGarbage collected after train\n")

x_dev, y_dev = data.getData(dataDir, configName, devSize, "Dev")
x_dev, y_dev = data.shuffleData(x_dev, y_dev)
y_dev_oh = data.oneHotY(y_dev)
#Mx_dev = data.stdData(x_dev, mean_pool, std_pool)
for index in range(5):
	plt.plot(plt_x, x_dev.iloc[index], 'r')
	figName = resultDir + "/" + configName + "/images_debug/" + modelName + "_dev_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_preStd_key" + str(y_dev.iloc[index].values[0]) + ".png"
	plt.savefig(figName)
	plt.close()

#Mx_dev = data.stdData(x_dev.to_numpy(), mean_pool, std_pool)
## Standardize
if (typeOfStd == "col"):
	x_dev = data.stdData(x_dev.to_numpy(), mean_pool, std_pool)
	x_dev = np.where(x_dev>10, 10, x_dev)
	x_dev = np.where(x_dev<-10, -10, x_dev)
elif(typeOfStd == "row"):
	x_dev = data.stdDataRowWise(x_dev.to_numpy())

## Commenting out the saving to file, because the error was not in data and it is huge is size
##Mdev_std_path = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/" + configName + "/dev_std_3p1_3p2.csv"
##Mnp.savetxt(dev_std_path, x_dev, delimiter=",")
for index in range(5):
	plt.plot(plt_x, x_dev[index], 'g')
	figName = resultDir + "/" + configName + "/images_debug/" + modelName + "_dev_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_postStd_key" + str(y_dev.iloc[index].values[0]) + ".png"
	plt.savefig(figName)
	plt.close()

gc.collect()
print("\nGarbage collected after dev\n")

##NOTE: Commenting because test is not required for re-running
#Mx_test, y_test = data.getData(dataDir, configName, 100, "Test")
#Mx_test, y_test = data.shuffleData(x_test, y_test)
#My_test_oh = data.oneHotY(y_test)
#M#Mx_test = data.stdData(x_test, mean_pool, std_pool)
#Mx_test = data.stdData(x_test.to_numpy(), mean_pool, std_pool)
gc.collect()
print("\nGarbage collected after test\n")

## Old way to load data, doesn't work for 1361 data
##MtrainData, devData, testData = classify_general.getData(dataDir, configName, trainSize, trainFlag, devFlag, testFlag)
##M
##Mx_train, y_train_oh = trainData
##Mx_dev, y_dev_oh = devData
##Mx_test, y_test_oh = testData

## Instantiate the model and test, dev and training sets
##resultDir = "/extra/manojgopale/AES_data/config3p1_15ktraining/result_new"
##modelName = "m_newscript"
np.random.seed()

numHiddenLayers = 3
actList = ['relu', 'relu', 'relu']
dropList = [0.1, 0.4, 0]
batchNorm = [0, 0, 0]
batchSize = 128
hiddenLayer = [64, 32, 256]
learingRate = np.float_power(10, np.random.random_integers(-3,0))
epsilonValue = np.float_power(10, np.random.random_integers(-7, 0))
allOpt = ['Adam', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']
#optStr = random.sample(allOpt, 1)
optStr = 'Adam'
learningRate = np.float_power(10,-3)
epsilonValue = np.float_power(10, -7)

runLogsPath = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/log/" + configName + "/allRuns.csv"
with open(runLogsPath, 'a') as f:
	## modelName must be unique like run_<someNum>
	f.write("\n%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s\n" %(modelName, numHiddenLayers, hiddenLayer, actList, dropList, batchNorm, batchSize, trainSize, learningRate, epsilonValue, optStr))

t0_time = time.time()
#classifier = classify_general.Classifier(resultDir, modelName, x_train, y_train_oh, x_dev, y_dev_oh, x_test, y_test_oh, hiddenLayer, actList, dropList, batchNorm)
## Taking 1361 power traces only
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
classifier.getPerplexity(x_dev[:,:numPowerTraces], y_dev_oh, configName)
