import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization
import pickle
import gzip
import pandas as pd
import numpy as np
import gc
import os
from datetime import date, datetime

from sklearn.utils import shuffle
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import clone_model
from keras.models import load_model

##Since the path to error_analysis file is already added to sys in run_3HL.py, we will directly 
##import those here

from error_analysis import errorAnalysis

## Commented for config5p4 to re-run
np.random.seed(9)

scaler = StandardScaler()

def process_inputs (dataPath):
	data = pd.read_csv(dataPath, header=None)
	dataShuffle = shuffle(data)
	x_data_shuffle = dataShuffle.iloc[:,0:-1]
	y_data = dataShuffle.iloc[:,-1]
	x_data = scaler.fit_transform(x_data_shuffle)
	return x_data, y_data

## dataPath : Path where the working directory is located. 
## config : The config for which the data needs to be loaded
## trainSize : Number of power traces per key to take
## testFlag : if true, then traina nd dev data won't be loaded
def getData(dataPath, config, trainSize, trainFlag, devFlag, testFlag):
	#runDir = "/extra/manojgopale/AES_data/config3p3_15ktraining/"
	runDir = dataPath
	dataDir = runDir + "/" + config +"/"
	devSize  = 1000
	testSize = 1000

	## Pre defining the arrays based on sizes of the data
	x_train = np.zeros((trainSize*4*64, 1361))
	x_dev = np.zeros((devSize*4*64, 1361))
	x_test = np.zeros((testSize*4*64, 1361))

	y_train = np.zeros((trainSize*4*64, 1))
	y_dev = np.zeros((devSize*4*64, 1))
	y_test = np.zeros((testSize*4*64, 1))

	for index, val in enumerate([1,0,3,2]):
		print("Started data processing for %d set\n" %(val))
		trainStr = dataDir + "aesData_config9_Train_" + str(val) + ".csv"
		devStr   = dataDir + "aesData_config9_Dev_" + str(val) + ".csv"
		testStr  = dataDir + "aesData_config9_Test_" + str(val) + ".csv"

		## Checking if the file size is 0, before processing data
		## This check is for cross config analysis, where traina nd dev are empty
		#if (os.stat(trainStr).st_size != 0):
		if (trainFlag):
			x_train_inter, y_train_inter = process_inputs(trainStr)
			## Substituing chunks of data to the allocated space in the array
			## The order of placement is 1,0,3,2 for the arrays
			x_train[trainSize*index*64: trainSize*(index+1)*64, : ] = x_train_inter
			y_train[trainSize*index*64: trainSize*(index+1)*64, 0] = y_train_inter
			print("Train= %s\n" %(trainFlag))
		else: 
			## Assigning the array's to 0's
			x_train[trainSize*index*64: trainSize*(index+1)*64, : ] = np.zeros((trainSize*64,1361))
			y_train[trainSize*index*64: trainSize*(index+1)*64, : ] = np.zeros((trainSize*64, 1))
			print("train= %s\n" %(trainFlag))

		#if (os.stat(devStr).st_size != 0):
		if (devFlag):
		## get the data for each sub part
			x_dev_inter, y_dev_inter     = process_inputs(devStr)
			x_dev[devSize*index*64: devSize*(index+1)*64, : ] = x_dev_inter
			y_dev[devSize*index*64: devSize*(index+1)*64, 0 ] = y_dev_inter
			print("Dev= %s\n" %(devFlag))
		else:
			x_dev[devSize*index*64: devSize*(index+1)*64, : ] = np.zeros((devSize*64, 1361))
			y_dev[devSize*index*64: devSize*(index+1)*64, : ] = np.zeros((devSize*64, 1))
			print("dev= %s\n" %(devFlag))

		## Test data is present so check is not performed
		if (testFlag):
			x_test_inter, y_test_inter   = process_inputs(testStr)
			x_test[testSize*index*64: testSize*(index+1)*64, : ] = x_test_inter
			y_test[testSize*index*64: testSize*(index+1)*64, 0 ] = y_test_inter
			print("Test= %s\n" %(testFlag))
		else:
			x_test[testSize*index*64: testSize*(index+1)*64, : ] = np.zeros((testSize*64, 1361))
			y_test[testSize*index*64: testSize*(index+1)*64, : ] = np.zeros((testSize*64, 1))
			print("test= %s\n" %(testFlag))

		print("Finished data processing for %d set\n" %(val))
	
	## Clear variables
	x_train_inter = None
	x_dev_inter = None
	x_test_inter = None
	y_train_inter = None
	y_dev_inter = None
	y_test_inter = None
	print("\nCleared variables\n")

	## One hot assignment
	n_classes = 256
	y_train_oh = np_utils.to_categorical(y_train, n_classes)
	y_dev_oh = np_utils.to_categorical(y_dev, n_classes)
	y_test_oh = np_utils.to_categorical(y_test, n_classes)
	
	print("\nOne-hot encoded for outputs\n")

	return (x_train, y_train_oh), (x_dev, y_dev_oh), (x_test, y_test_oh)

class Classifier:
	def __init__(self, resultDir: str, modelName: str, x_train, y_train_oh, x_dev, y_dev_oh, x_test, y_test_oh, hiddenLayer, actList, dropList, batchNorm, numPowerTraces):
		""" Initialize parameters and sequential model for training
		"""
		self.resultDir = resultDir
		self.modelName = modelName
		self.x_train = x_train[:,0:numPowerTraces]
		self.x_dev = x_dev[:, 0:numPowerTraces]
		self.x_test = x_test[:, 0:numPowerTraces]
		self.y_train_oh = y_train_oh
		self.y_dev_oh = y_dev_oh
		self.y_test_oh = y_test_oh

		self.hiddenLayer = hiddenLayer
		self.actList = actList
		self.dropList = dropList
		self.batchNorm = batchNorm
		self.numPowerTraces = numPowerTraces
		
		self.model = Sequential()

		## Building models
		for index, layer in enumerate(hiddenLayer):
			if (index == 0):
				## 1st layer
				self.model.add(Dense(hiddenLayer[index], activation=actList[index], input_shape=(numPowerTraces,)))
			else:
				self.model.add(Dense(hiddenLayer[index], activation=actList[index]))

			if (batchNorm[index]):
				self.model.add(BatchNormalization())
			
			self.model.add(Dropout(dropList[index]))

		## Last layer is softmax layer
		self.model.add(Dense(256, activation='softmax'))

		self.model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')
		print("Model summary\n")
		print(self.model.summary())

	def train(self, batchSize):
		""" Train the model with the training data
		batchSize : batch size during trainig
		"""

		Epochs = 100

		logFile = self.resultDir + '/' + self.modelName + '_' + str(batchSize) +'.log'
		csv_logger = CSVLogger(logFile, append=True, separator="\t")
		
		earlyStop = EarlyStopping(monitor='val_categorical_accuracy', patience=5, mode='auto', verbose=1, restore_best_weights=True)
		
		##filePath = self.resultDir + '/' + self.modelName + '_checkPoint_best_model.hdf5'
		#### This file will include the epoch number when it gets saved.
		##repeatingFile = self.resultDir + '/' + self.modelName +'_{epoch:02d}_epoch_acc_{accVar:.2f}.hdf5'
		#### By default the every_10epochs will save the model at every 10 epochs
		##checkPoint = newCallBacks.ModelCheckpoint_every_10epochs(filePath, repeatingFile, self.x_test, self.y_test_oh , monitor='val_categorical_accuracy', verbose=1, save_best_only=True, every_10epochs=True)
		
		self.history = self.model.fit(self.x_train, self.y_train_oh, batch_size= batchSize, epochs=Epochs, verbose=1, shuffle= True, validation_data=(self.x_dev, self.y_dev_oh), callbacks=[csv_logger, earlyStop])

	def evaluate(self):
		""" Evaluate the model on itself
		"""

		## We should be evaluating on dev dataset as well, so commenting x_test
		#self.model_score = self.model.evaluate(self.x_test, self.y_test_oh, batch_size=2048)
		self.model_score = self.model.evaluate(self.x_dev, self.y_dev_oh, batch_size=2048)
		print("%s score = %f\n" %(self.modelName, self.model_score[1]))

		##Saving atucal vs predicted predictions
		##np.argmax returns the index where it see's 1 in the row
		#y_pred = np.argmax(self.model.predict(self.x_test, batch_size=2048), axis=1)
		y_pred = np.argmax(self.model.predict(self.x_dev, batch_size=2048), axis=1)

		## vstack will stack them in 2 rows, so we use Trasnpose to get them in column stack
		#output_predict = np.vstack((np.argmax(self.y_test_oh, axis=1), y_pred)).T
		output_predict = np.vstack((np.argmax(self.y_dev_oh, axis=1), y_pred)).T

		self.outputFile = self.resultDir + '/' + self.modelName + '_' + str(len(self.hiddenLayer)) + 'HL_'  + str(self.history.epoch[-1]+1) + 'epochs_' + '{0:.2f}'.format(self.model_score[1]*100).replace('.', 'p') + '_acc_' + "outputPredict.csv" 
		
		np.savetxt(self.outputFile, output_predict, fmt="%5.0f", delimiter=",")

		##Error Analysis of the prediction
		errorAnalysis(self.outputFile)

		return self.model_score

	def saveModel(self):
		""" Save the model
		"""
		saveStr = self.resultDir + '/' + self.modelName + '_' + str(len(self.hiddenLayer)) + 'HL_'  + str(self.history.epoch[-1]+1) + 'epochs_' + '{0:.2f}'.format(self.model_score[1]*100).replace('.', 'p') + '_acc_' + '.h5'
		print("Saving model to\n%s\n" %(saveStr))
		self.model.save(saveStr)

	def keyAccuracy(self, config):

		df = pd.read_csv(self.outputFile, header=None)
		
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
		saveFile = self.resultDir + '/' + self.modelName + '_' + str(len(self.hiddenLayer)) + 'HL_'  + str(self.history.epoch[-1]+1) + 'epochs_' + '{0:.2f}'.format(self.model_score[1]*100).replace('.', 'p') + '_acc_' + "keyAccuracy.tsv" 
		keyAcc.to_csv(saveFile, sep='\t', header=True, index=False)
		
		##Checking for entopy calculations of the predictions
		pred = self.model.predict(self.x_dev, batch_size=2048)
		
		## np.argsort(-pred) gets the order in which the indexes will be arranged 
		## np.argsort() of above will return the rank of the indexes with their values in the coressponding indexes
		rank = np.argsort(np.argsort(-pred))
		
		## Get the actual predictions from y_dev, need to convert one-hot to actual numbers
		dev_actual = np.argmax(self.y_dev_oh, axis=1)
		
		## Get the prediction ranks for each prediction
		prediction_ranks = rank[np.arange(len(dev_actual)), dev_actual]
		
		## getting the mean will also get the accuracy for the recall_at
		## recall = 1 , will get you accuracy at one shot
		recall_1 = np.mean(prediction_ranks < 1)
		recall_10 = np.mean(prediction_ranks < 10)
		recall_25 = np.mean(prediction_ranks < 25)
		recall_40 = np.mean(prediction_ranks < 40)
		recall_50 = np.mean(prediction_ranks < 50)
	
		print("model= %s\nrecall_1= %s\nrecall_10= %s\nrecall_25= %s\nrecall_40= %s\nrecall_50= %s" %(self.modelName, recall_1, recall_10, recall_25, recall_40, recall_50))

		## Create confusion matriux for aggregated computation
		conf = confusion_matrix(df[0], df[1])
		## Get the index for each row's max value. This is the column number in each row where the max value is located
		rowArgMax = np.argmax(conf, axis=1)

		logFile = self.resultDir + '../../log/' + config + '/' + self.modelName + '_' + str(len(self.hiddenLayer)) + 'HL_'  + str(self.history.epoch[-1]+1) + 'epochs_' + '{0:.2f}'.format(self.model_score[1]*100).replace('.', 'p') + '_acc_' + "keyAccuracy.log" 

		with open(logFile, 'a') as f:
			f.write("model= %s\nrecall_1= %s\nrecall_10= %s\nrecall_25= %s\nrecall_40= %s\nrecall_50= %s\n\n" %(self.modelName, recall_1, recall_10, recall_25, recall_40, recall_50))
			for row in range(256):
				if (row != rowArgMax[row]):
					print("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)---" %(row, conf[row, row]/10, rowArgMax[row], conf[row, rowArgMax[row]]/10))
					f.write("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)---\n" %(row, conf[row, row]/10, rowArgMax[row], conf[row, rowArgMax[row]]/10))
				else:
					print("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)" %(row, conf[row, row]/10, rowArgMax[row], conf[row, rowArgMax[row]]/10))
					f.write("key= %s (%s%% acc), maxPredKey= %s (%s%% acc)\n" %(row, conf[row, row]/10, rowArgMax[row], conf[row, rowArgMax[row]]/10))


