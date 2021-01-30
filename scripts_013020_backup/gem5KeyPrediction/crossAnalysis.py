import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from collections import OrderedDict
import tensorflow as tf

import sys
sys.path.insert(0, '/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/')
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
parser.add_option('--trainFlag',
									action = 'store', type='int', dest='trainFlag', default = 1)
parser.add_option('--devFlag',
									action = 'store', type='int', dest='devFlag', default = 1)
parser.add_option('--testFlag',
									action = 'store', type='int', dest='testFlag', default = 0)
parser.add_option('--count',
									action = 'store', type='int', dest='count', default = 0)

(options, args) = parser.parse_args()

########

devDataConfig  = options.devDataConfig
trainSize   = options.trainSize
modelDir 		= options.modelDir
devDataDir 		= options.devDataDir
trainFlag 	= options.trainFlag
devFlag 		= options.devFlag
testFlag 		= options.testFlag
count   		= options.count

########

### TODO:
### 1. Load dev data 
### 2. Load models one-by-one and evaluate
### 3. Create csv/tsv files and run error_analysis on it

devDir = devDataDir + devDataConfig + "/"
_, devData, _ = classify_general.getData(devDataDir, devDataConfig, trainSize, trainFlag, devFlag, testFlag)
x_dev, y_dev_oh = devData

modelDict = OrderedDict()
modelDict = {\
"config3p1": "rerun_cw_run_177_4HL_11epochs_100p00_acc_.h5",\
"config3p2": "rerun_cw_run_config3p2_177_4HL_11epochs_98p05_acc_.h5",\
"config3p3": "rerun_cw_run_177_4HL_11epochs_98p05_acc_.h5",\
"config3p4": "rerun_cw_run_config3p4_177_4HL_17epochs_100p00_acc_.h5",\
"config4p1": "rerun_cw_run_177_4HL_11epochs_97p66_acc_.h5",\
"config4p2": "rerun_cw_run_177_4HL_11epochs_97p66_acc_.h5",\
"config4p3": "rerun_cw_run_config4p3_177_4HL_17epochs_100p00_acc_.h5",\
"config4p4": "rerun_cw_run_177_4HL_11epochs_98p05_acc_.h5",\
"config5p1": "rerun_cw_run_177_4HL_18epochs_100p00_acc_.h5",\
"config5p2": "rerun_cw_run_config5p2_177_4HL_18epochs_100p00_acc_.h5",\
"config5p3": "rerun_cw_run_177_4HL_11epochs_97p66_acc_.h5",\
"config5p4": "rerun_cw_run_177_4HL_11epochs_97p66_acc_.h5",\
}
configList = [\
"config3p1", "config3p2", "config3p3", "config3p4", \
"config4p1", "config4p2", "config4p3", "config4p4", \
"config5p1", "config5p2", "config5p3", "config5p4", \
]

modelList = [\
"rerun_cw_run_177_4HL_11epochs_100p00_acc_.h5",\
"rerun_cw_run_config3p2_177_4HL_11epochs_98p05_acc_.h5",\
"rerun_cw_run_177_4HL_11epochs_98p05_acc_.h5",\
"rerun_cw_run_config3p4_177_4HL_17epochs_100p00_acc_.h5",\
"rerun_cw_run_177_4HL_11epochs_97p66_acc_.h5",\
"rerun_cw_run_177_4HL_11epochs_97p66_acc_.h5",\
"rerun_cw_run_config4p3_177_4HL_17epochs_100p00_acc_.h5",\
"rerun_cw_run_177_4HL_11epochs_98p05_acc_.h5",\
"rerun_cw_run_177_4HL_18epochs_100p00_acc_.h5",\
"rerun_cw_run_config5p2_177_4HL_18epochs_100p00_acc_.h5",\
"rerun_cw_run_177_4HL_11epochs_97p66_acc_.h5",\
"rerun_cw_run_177_4HL_11epochs_97p66_acc_.h5",\
]

for modelConfig, modelName in modelDict.items():
	modelPath = modelDir + configList[count] + "/" + modelList[count]
	model = load_model(modelPath)
	print("Loaded model from\n%s" %(modelPath))

	## Evaluate the performance of model on testData
	model_score = model.evaluate(x_dev, y_dev_oh, batch_size=2048)
	print("\nmodel of: %s score on devData: %s is: %s\n" %(configList[count], devDataConfig, model_score[1]))
	
	## Convert from one-hot to numerical prediction
	y_pred = np.argmax(model.predict(x_dev, batch_size=2048), axis=1)
	
	## vstack the actual and predicted output and take transpose
	output_predict = np.vstack((np.argmax(y_dev_oh, axis=1), y_pred)).T
	
	## Save it to csv file for future analysis
	outputFile = modelDir + "/" + devDataConfig + "/"  + "dataOf_" + devDataConfig + "_modelOf_" + configList[count] + "_dev.csv" 
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
	saveFile = modelDir + "/" + devDataConfig + "/"  + "dataOf_" + devDataConfig + "_modelOf_" + configList[count] + "_dev_keyAccuracy.tsv" 
	keyAcc.to_csv(saveFile, sep='\t', header=True, index=False)
	
	##Checking for entopy calculations of the predictions
	pred = model.predict(x_dev, batch_size=2048)
	
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
	
	print("model= %s\nrecall_1= %s\nrecall_10= %s\nrecall_25= %s\nrecall_40= %s\nrecall_50= %s" %(modelList[count], recall_1, recall_10, recall_25, recall_40, recall_50))
	
	## Create confusion matriux for aggregated computation
	conf = confusion_matrix(df[0], df[1])
	## Get the index for each row's max value. This is the column number in each row where the max value is located
	rowArgMax = np.argmax(conf, axis=1)
	
	logFile = modelDir + '/../log/' + devDataConfig + '/dataOf_' + devDataConfig + '_modelOf_'+ configList[count] + "_" + '{0:.2f}'.format(model_score[1]*100).replace('.', 'p') + '_acc_' + "keyAccuracy.log" 
	
	with open(logFile, 'a') as f:
		f.write("model= %s\nrecall_1= %s\nrecall_10= %s\nrecall_25= %s\nrecall_40= %s\nrecall_50= %s\n\n" %(modelList[count], recall_1, recall_10, recall_25, recall_40, recall_50))
		for row in range(256):
			if (row != rowArgMax[row]):
				print("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)---" %(row, conf[row, row]/10, rowArgMax[row], conf[row, rowArgMax[row]]/10))
				f.write("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)---\n" %(row, conf[row, row]/10, rowArgMax[row], conf[row, rowArgMax[row]]/10))
			else:
				print("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)" %(row, conf[row, row]/10, rowArgMax[row], conf[row, rowArgMax[row]]/10))
				f.write("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)\n" %(row, conf[row, row]/10, rowArgMax[row], conf[row, rowArgMax[row]]/10))

	count = count + 1
	if (count %4 == 0):
		break

