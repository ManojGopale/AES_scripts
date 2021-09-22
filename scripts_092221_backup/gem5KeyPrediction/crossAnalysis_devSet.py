##NOTE: Change file to match crossAnalysis_testSet.py
import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
import tensorflow as tf
import time
import os
import keras
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/')
import newLoadData
import gc
import classify_general
from error_analysis import errorAnalysis

from optparse import OptionParser

parser = OptionParser()

parser.add_option('--modelDir',
									action = 'store', type='string', dest='modelDir', default = '/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/')
parser.add_option('--devDataDir',
									action = 'store', type='string', dest='devDataDir', default = '/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/')
parser.add_option('-t','--devDataConfig',
									action = 'store', type='string', dest='devDataConfig', default = 'config3p1')
parser.add_option('--trainSize',
									action = 'store', type='int', dest='trainSize', default = 28000)
parser.add_option('--devSize',
									action = 'store', type='int', dest='devSize', default = 1000)
parser.add_option('--trainFlag',
									action = 'store', type='int', dest='trainFlag', default = 1)
parser.add_option('--devFlag',
									action = 'store', type='int', dest='devFlag', default = 1)
parser.add_option('--testFlag',
									action = 'store', type='int', dest='testFlag', default = 0)
parser.add_option('--inputTraces',
									action = 'store', type='int', dest='inputTraces', default = 1500)
parser.add_option('--typeOfStd',
									action = 'store', type='string', dest='typeOfStd', default = 'col')

(options, args) = parser.parse_args()

########

devDataConfig  = options.devDataConfig
trainSize      = options.trainSize
devSize        = options.devSize
modelDir 		   = options.modelDir
devDataDir 	   = options.devDataDir
trainFlag 	   = options.trainFlag
devFlag 		   = options.devFlag
testFlag 		   = options.testFlag
inputTraces    = options.inputTraces ##input length for the model
typeOfStd      = options.typeOfStd ##type of standardization to use, some diff in the way we standardize

########

### TODO:
### 1. Dev data for 1 or 2 configs will be loaded
### 2. Models of 5p3_15000TrainSize run's will be loaded to see the accuracy on them
### 3. Once we get a good set on the dev data, we can try it on other configs as well

## NOTE: CHECK THE CORRECT PATHS FOR MEAN AND STD DEV
if (typeOfStd == "col"):
	## Load meand and std dev 
	## Added this to classify_general file
	meanPath = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/allConfigs/config3p1_3p2_3p3_3p4_4p1_4p2_4p3_4p4_5p1_5p2_5p3_5p4_mean.csv"
	stdPath = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/allConfigs/config3p1_3p2_3p3_3p4_4p1_4p2_4p3_4p4_5p1_5p2_5p3_5p4_std.csv"
	## Converting them to numpy array for standardisation
	mean_pool = pd.read_csv(meanPath, header=None).to_numpy()
	std_pool = pd.read_csv(stdPath, header=None).to_numpy()
	print("Loaded mean and std files from\n%s\n%s\n" %(meanPath, stdPath))
	## Reshaping so that it matches the standardization function and not error out
	mean_pool = mean_pool.reshape(mean_pool.shape[0], )
	std_pool = std_pool.reshape(std_pool.shape[0], )
	

#MdevDir = devDataDir + devDataConfig + "/"
devDir = devDataDir + "/"
data=newLoadData.Data()
## NOTE:
## With new method of standardization directly loading mean and ,stdDev, we do not need to import train dataset now
if (typeOfStd == "selfNorm"):
	x_train, y_train = data.getData(devDir, devDataConfig, trainSize, "Train")
	x_train, y_train = data.shuffleData(x_train, y_train)
	y_train_oh = data.oneHotY(y_train)
	x_train_mean, x_train_std = data.getStdParam(x_train)
	mean_pool, std_pool = data.getStdParam(x_train) ## Comment after 1361 run is over
	mean_pool = mean_pool.to_numpy()
	mean_pool = mean_pool.reshape(mean_pool.shape[0], )
	std_pool = std_pool.to_numpy()
	std_pool = std_pool.reshape(mean_pool.shape[0], )
	## Comment till here from NOTE after 1361 runs are done

## Write the mean and std to csv for future use
#MmeanPath = devDir + "/" + devDataConfig + "/" + modelName + "_mean.csv"
#MstdPath = devDir + "/" + devDataConfig + "/" + modelName + "_std.csv"
#Mx_train_mean.to_csv(meanPath, index=False, header=False)
#Mx_train_std.to_csv(stdPath, index=False, header=False)
#Mx_train = data.stdData(x_train, x_train_mean, x_train_std)
##Mgc.collect()
##Mprint("\nGarbage collected after train\n")

x_dev, y_dev = data.getData(devDir, devDataConfig, devSize, "Dev")
x_dev, y_dev = data.shuffleData(x_dev, y_dev)
y_dev_oh = data.oneHotY(y_dev)
#Mx_dev = data.stdData(x_dev, x_train_mean, x_train_std)[:,:inputTraces]
plt_x = np.linspace(0,inputTraces-1, num=inputTraces)
for index in range(5):
	plt.plot(plt_x, x_dev.iloc[index,:inputTraces], 'r')
	figName = modelDir + "/" + devDataConfig + "/images_debug/hyper_dev_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_preStd_key" + str(y_dev.iloc[index].values[0]) + ".png"
	plt.savefig(figName)
	plt.close()

if(typeOfStd == "selfNorm"):
	x_dev = data.stdData(x_dev.to_numpy(), mean_pool, std_pool)
elif (typeOfStd=="col"):
	x_dev = data.stdData(x_dev.to_numpy(), mean_pool, std_pool)
	##NOTE: Clipping used only for poolAll datasets
	print("\nInside col std.\n")
	x_dev = np.where(x_dev>10, 10, x_dev)
	x_dev = np.where(x_dev<-10, -10, x_dev)
elif (typeOfStd=="row"):
	x_dev = data.stdDataRowWise(x_dev.to_numpy())

for index in range(5):
	plt.plot(plt_x, x_dev[index,:inputTraces], 'g')
	figName = modelDir + "/" + devDataConfig + "/images_debug/hyper_dev_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_postStd_key" + str(y_dev.iloc[index].values[0]) + ".png"
	plt.savefig(figName)
	plt.close()

gc.collect()
print("\nGarbage collected after dev\n")

##M_, devData, _ = classify_general.getData(devDataDir, devDataConfig, trainSize, trainFlag, devFlag, testFlag)
##Mx_dev, y_dev_oh = devData

modelDict = OrderedDict()
modelDict = {\
"config3p1": "rerun_dataEns49_43_config_tr7500_3p1_3HL_6epochs_100p00_acc_.h5",\
"config3p2": "rerun_dataEns49_43_config_tr7500_3p2_3HL_7epochs_100p00_acc_.h5",\
"config3p3": "rerun_dataEns49_43_config_tr7500_3p3_3HL_6epochs_100p00_acc_.h5",\
"config3p4": "rerun_dataEns49_43_config_tr7500_3p4_3HL_9epochs_100p00_acc_.h5",\
"config4p1": "rerun_dataEns49_43_config_tr7500_4p1_3HL_6epochs_100p00_acc_.h5",\
"config4p2": "rerun_dataEns49_43_config_tr7500_4p2_3HL_6epochs_99p99_acc_.h5",\
"config4p3": "rerun_dataEns49_43_config_tr7500_4p3_3HL_8epochs_100p00_acc_.h5",\
"config4p4": "rerun_dataEns49_43_config_tr7500_4p4_3HL_6epochs_100p00_acc_.h5",\
"config5p1": "rerun_dataEns49_43_config_tr7500_5p1_3HL_7epochs_100p00_acc_.h5",\
"config5p2": "rerun_dataEns49_43_config_tr7500_5p2_3HL_6epochs_100p00_acc_.h5",\
"config5p3": "rerun_dataEns49_43_config_tr7500_5p3_3HL_6epochs_100p00_acc_.h5",\
"config5p4": "rerun_dataEns49_43_config_tr7500_5p4_3HL_6epochs_100p00_acc_.h5"\
}


#M## Models from the run's for config5p3_15000TrainSize config
#MmodelList = [\
#M"rerun49_43_S7_10_3HL_77epochs_99p93_acc_.h5",\
#M"rerun49_43_S7_20_3HL_43epochs_100p00_acc_.h5",\
#M"rerun49_43_S7_50_3HL_21epochs_100p00_acc_.h5",\
#M"rerun49_43_S7_100_3HL_21epochs_100p00_acc_.h5",\
#M"rerun49_43_S7_200_3HL_18epochs_100p00_acc_.h5",\
#M"rerun49_43_S7_500_3HL_13epochs_100p00_acc_.h5",\
#M"rerun49_43_S7_1000_3HL_17epochs_100p00_acc_.h5",\
#M"rerun49_43_S7_2000_3HL_14epochs_100p00_acc_.h5",\
#M"rerun49_43_S7_2500_3HL_17epochs_99p99_acc_.h5",\
#M"rerun49_43_S7_3000_3HL_13epochs_99p99_acc_.h5"\
#M]

for modelConfig, modelName in modelDict.items():
	modelPath = modelDir + "/" + modelConfig + "/" + modelName
	## clear_Session helps the model to clear so that we don't get the exception after loading 5 models
	keras.backend.clear_session()
	if (os.path.isfile(modelPath)):
		model = load_model(modelPath)
		print("Loaded model from\n%s" %(modelPath))

		## Evaluate the performance of model on testData
		model_score = model.evaluate(x_dev[:,:inputTraces], y_dev_oh, batch_size=256)
		print("\n%s model of: %s score on devData: %s is: %s\n" %(modelConfig, modelName, devDataConfig, model_score[1]))
		
		## Convert from one-hot to numerical prediction
		y_pred = np.argmax(model.predict(x_dev[:,:inputTraces], batch_size=256), axis=1)
		
		## vstack the actual and predicted output and take transpose
		output_predict = np.vstack((np.argmax(y_dev_oh, axis=1), y_pred)).T
		
		## Save it to csv file for future analysis
		## Split the modelName so that it has config and run number in the name
		outputFile = modelDir + "/" + devDataConfig + "/"  + "dataOf_" + devDataConfig + "_modelOf_" + "_".join(modelName.split("_")[1:2]) + "_dev.csv" 
		np.savetxt(outputFile, output_predict, fmt="%5.0f", delimiter=",")
		
		##Error Analysis
		errorAnalysis(outputFile)

		df = pd.read_csv(outputFile, header=None)
		
		error_df = df[df[0]!=df[1]].astype('category')
		error_df[2] = error_df[0].astype(str).str.cat(error_df[1].astype(str), sep="-")
		
		totalCount = df[0].count()
		errorCount = error_df[2].count()
		accuracy = ((df[0].count()-error_df[2].count())/df[0].count())*100
		
		
		## to get the accuracy of individual keys, we need to count the number of rows in error_df for the same key
		## dubtract it from total data elements and divide it by the total number of data elements for each.
		
		## For example
		#c1 = df[0][df[0] == 0].count()
		#error_0 = error_df[0][error_df[0]==0].count()
		#
		#acc_0 = ((c1-error_0)/c1)*100
		
		## Now to loop it
		keyAcc = pd.DataFrame(columns={'key', 'acc'})
		
		for key in range(256):
			totalKey = df[0][df[0]==key].count()
			keyErrors = error_df[0][error_df[0]==key].count()
			acc = ((totalKey-keyErrors)/totalKey)*100
			keyAcc.loc[key] = {'key': key, 'acc': acc}
			#print("key= %s, acc= %s" %(key, acc))
		
		## Save to tsv
		saveFile = modelDir + "/" + devDataConfig + "/"  + "dataOf_" + devDataConfig + "_modelOf_" + "_".join(modelName.split("_")[1:2]) + "_dev_keyAccuracy.tsv" 
		keyAcc.to_csv(saveFile, sep='\t', header=True, index=False)
		
		##Checking for entopy calculatirerun_prev_rowStd_config3p1_22_4HL_15epochs_99p22_acc_.h5"\ons of the predictions
		pred = model.predict(x_dev[:,:inputTraces], batch_size=256)
		
		## np.argsort(-pred) gets the order in which the indexes will be arranged 
		## np.argsort() of above will return the rank of the indexes with their values in the coressponding indexes
		rank = np.argsort(np.argsort(-pred))
		
		## Get the actual predictions from y_dev, need to convert one-hot to actual numbers
		dev_actual = np.argmax(y_dev_oh, axis=1)
		
		## Get the prediction ranks for each prediction
		prediction_ranks = rank[np.arange(len(dev_actual)), dev_actual]
		
		## getting the mean will also get the accuracy for the recall_at
		## recall = 1 , will get you accuracy at one shot
		recall_1 = np.mean(prediction_ranks < 1)
		recall_10 = np.mean(prediction_ranks < 10)
		recall_25 = np.mean(prediction_ranks < 25)
		recall_40 = np.mean(prediction_ranks < 40)
		recall_50 = np.mean(prediction_ranks < 50)
		
		print("model= %s\nrecall_1= %s\nrecall_10= %s\nrecall_25= %s\nrecall_40= %s\nrecall_50= %s" %(modelName, recall_1, recall_10, recall_25, recall_40, recall_50))
		
		## Create confusion matriux for aggregated computation
		conf = confusion_matrix(df[0], df[1])
		## Get the index for each row's max value. This is the column number in each row where the max value is located
		rowArgMax = np.argmax(conf, axis=1)
		
		logFile = modelDir + '/../log/' + devDataConfig + '/dataOf_' + devDataConfig + '_modelOf_'+ "_".join(modelName.split("_")[1:2]) + "_" + '{0:.2f}'.format(model_score[1]*100).replace('.', 'p') + '_acc_' + "keyAccuracy.log" 
		
		## Divisor so that the % is 100%.
		## 1000 devSize -> 1 divisior
		##  devSize -> (devSize) / 1000
		divisor = (devSize)/1000
		with open(logFile, 'a') as f:
			f.write("model= %s\nrecall_1= %s\nrecall_10= %s\nrecall_25= %s\nrecall_40= %s\nrecall_50= %s\n\n" %(modelName, recall_1, recall_10, recall_25, recall_40, recall_50))
			for row in range(256):
				if (row != rowArgMax[row]):
					print("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)---" %(row, conf[row, row]/divisor, rowArgMax[row], conf[row, rowArgMax[row]]/divisor))
					f.write("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)---\n" %(row, conf[row, row]/divisor, rowArgMax[row], conf[row, rowArgMax[row]]/divisor))
				else:
					print("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)" %(row, conf[row, row]/divisor, rowArgMax[row], conf[row, rowArgMax[row]]/divisor))
					f.write("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)\n" %(row, conf[row, row]/divisor, rowArgMax[row], conf[row, rowArgMax[row]]/divisor))

		## Get perplexity
		pred = model.predict(x_dev[:,:inputTraces], batch_size=2048)
		
		dev_actual_perp = np.argmax(y_dev_oh, axis=1)
		dev_actual_perp = dev_actual_perp.reshape(dev_actual_perp.shape[0], 1) #shape should be in array form for take_along_axis
		
		pred_prob = np.take_along_axis(pred, dev_actual_perp, 1)
		
		## loop to calculate perplexity
		product = 1
		N = pred_prob.shape[0] #Total number of dev samples
		for index in range(N):
			## Take the n'th root of each value and then multiply them
			product = product * np.power(pred_prob[index], (1/N))
		
		perplexity = 1/product
		print("perplexity of model=%s on devData of %s is %s\n" %(modelName, devDataConfig, perplexity))
