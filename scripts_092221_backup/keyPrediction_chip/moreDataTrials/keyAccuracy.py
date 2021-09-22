import pandas as pd
import numpy as np

from keras.models import load_model

import sys
sys.path.insert(0, '/xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/scr/')
import classify_3HL

## load test data
## This can be done standalone in the script that call this function
#dataPath = "/xdisk/rlysecky/manojgopale/extra/chipWhisperer_data/trace_key_1500/"
#(_, _), (x_dev, y_dev), (_, _) = classify_3HL.getData(dataPath, 15000, 0, 1, 0) 

##TODO x_dev and y_dev must be loaded before calling this function
## resultDir -> Where results, predOutput, keyAccuracies per key will be saved to
## modelDir -> Where the models are saved
## logDir -> Where recall accuracies will be stored to a file
## modelName -> Name of the model used to find accuracies for
## savePrefix -> prefix to add while saving the tsv and other files
## x_dev, y_dev -> Files to be tested against

def keyAccuracy(x_dev, y_dev, modelDir, modelName, savePrefix, resultDir, logDir):
	modelStr = modelDir + "/" + modelName
	
	### load model
	model = load_model(modelStr)
	
	## get predictions
	y_pred = np.argmax(model.predict(x_dev, batch_size=2048), axis=1)
	
	## vstack will stack them in 2 rows, so we use Trasnpose to get them in column stack
	output_predict = np.vstack((np.argmax(y_dev, axis=1), y_pred)).T
	
	## Save to csv for further analysis
	outputFile = resultDir + "/" + savePrefix + "_predOutput.csv"
	
	np.savetxt(outputFile, output_predict, fmt="%5.0f", delimiter=",")
	
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
	
	for key in range(255):
		totalKey = df[0][df[0]==key].count()
		keyErrors = error_df[0][error_df[0]==key].count()
		acc = ((totalKey-keyErrors)/totalKey)*100
		keyAcc.loc[key] = {'key': key, 'acc': acc}
		#print("key= %s, acc= %s" %(key, acc))
	
	## Save to csv
	saveFile=resultDir + "/" + savePrefix + "_keyAccuracy.tsv"
	keyAcc.to_csv(saveFile, sep='\t', header=True, index=False)
	
	##Checking for entopy calculations of the predictions
	pred = model.predict(x_dev, batch_size=2048)
	
	## np.argsort(-pred) gets the order in which the indexes will be arranged 
	## np.argsort() of above will return the rank of the indexes with their values in the coressponding indexes
	rank = np.argsort(np.argsort(-pred))
	
	## Get the actual predictions from y_dev, need to convert one-hot to actual numbers
	dev_actual = np.argmax(y_dev, axis=1)
	
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

	logFile = logDir + "/recall_3HL_moreData.log"

	with open(logFile, 'a') as f:
		f.write("\nmodel= %s\nrecall_1= %s\nrecall_10= %s\nrecall_25= %s\nrecall_40= %s\nrecall_50= %s\n" %(modelName, recall_1, recall_10, recall_25, recall_40, recall_50))


