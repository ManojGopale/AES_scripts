import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping
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
from keras.models import clone_model
from keras.models import load_model

##Since the path to error_analysis file is already added to sys in run_3HL.py, we will directly 
##import those here

from error_analysis import errorAnalysis

## Commented for config5p4 to re-run
np.random.seed(9)

scaler = StandardScaler()

def process_inputs (dataPath):
	data = pd.read_pickle(dataPath)
	## Data is already shuffled during saving
	#dataShuffle = shuffle(data)
	## Apply to get the 1'st element of the entire key
	y_data = data.key.apply(lambda x: x[0]).values
	## For x_data, awe need to first convert the list of memmap array to separate columns of numbers .apply(pd.Series) does that
	## .values converts it to an array
	## then we apply scaler.fir_trasnform to it
	#x_data = scaler.fit_transform(data.trace.apply(pd.Series).values)

	## The fit_transform is not doing -1 and 1 scaling, we will run it without doing any transformation
	x_data = data.trace.apply(pd.Series).values

	## Without transformation
	##MG x_data = data.trace.apply(pd.Series).values
	return x_data, y_data

## dataPath : Path where the working directory is located. 
## trainSize : Number of power traces per key to take
## testFlag : if true, then traina nd dev data won't be loaded
def getData(dataPath, trainSize, trainFlag, devFlag, testFlag):
	##TODO changed devSize to 1000 for trails for more trainSize data. It was 5000 for default run
	devSize  = 5000
	testSize = 1000
	numTraces = 1500 ##Number of traces collected per key

	## Pre defining the arrays based on sizes of the data
	x_train = np.zeros((trainSize*256, numTraces))
	x_dev = np.zeros((devSize*256, numTraces))
	x_test = np.zeros((testSize*256, numTraces))

	y_train = np.zeros((trainSize*256, 1))
	y_dev = np.zeros((devSize*256, 1))
	y_test = np.zeros((testSize*256, 1))

	for index, val in enumerate(range(0,256)):
		print("Started data processing for %d key\n" %(val))
		trainStr = dataPath + "train_" + str(val) + ".pkl.zip"
		devStr   = dataPath + "dev_" + str(val) + ".pkl.zip"
		testStr  = dataPath + "test_" + str(val) + ".pkl.zip"

		## Checking if the file size is 0, before processing data
		## This check is for cross config analysis, where traina nd dev are empty
		#if (os.stat(trainStr).st_size != 0):
		if (trainFlag):
			x_train_inter, y_train_inter = process_inputs(trainStr)
			## Substituing chunks of data to the allocated space in the array
			## The order of placement is 1,0,3,2 for the arrays
			x_train[trainSize*index: trainSize*(index+1), : ] = x_train_inter
			y_train[trainSize*index: trainSize*(index+1), 0] = y_train_inter
			print("Train= %s\n" %(trainFlag))
		else: 
			## Assigning the array's to 0's
			x_train[trainSize*index: trainSize*(index+1), : ] = np.zeros((trainSize,numTraces))
			y_train[trainSize*index: trainSize*(index+1), : ] = np.zeros((trainSize, 1))
			print("train= %s\n" %(trainFlag))

		#if (os.stat(devStr).st_size != 0):
		if (devFlag):
		## get the data for each sub part
			x_dev_inter, y_dev_inter     = process_inputs(devStr)
			x_dev[devSize*index: devSize*(index+1), : ] = x_dev_inter
			y_dev[devSize*index: devSize*(index+1), 0 ] = y_dev_inter
			print("Dev= %s\n" %(devFlag))
		else:
			x_dev[devSize*index: devSize*(index+1), : ] = np.zeros((devSize, numTraces))
			y_dev[devSize*index: devSize*(index+1), : ] = np.zeros((devSize, 1))
			print("dev= %s\n" %(devFlag))

		## Test data is present so check is not performed
		if (testFlag):
			x_test_inter, y_test_inter   = process_inputs(testStr)
			x_test[testSize*index: testSize*(index+1), : ] = x_test_inter
			y_test[testSize*index: testSize*(index+1), 0 ] = y_test_inter
			print("Test= %s\n" %(testFlag))
		else:
			x_test[testSize*index: testSize*(index+1), : ] = np.zeros((testSize, numTraces))
			y_test[testSize*index: testSize*(index+1), : ] = np.zeros((testSize, 1))
			print("test= %s\n" %(testFlag))

		print("Finished data processing for %d key\n" %(val))
	
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
	def __init__(self, resultDir: str, modelName: str, x_train, y_train_oh, x_dev, y_dev_oh, x_test, y_test_oh, drop1, drop2, drop3, drop4, drop5, drop6, drop7):
		""" Initialize parameters and sequential model for training
		"""
		self.resultDir = resultDir
		self.modelName = modelName
		self.x_train = x_train
		self.x_dev = x_dev
		self.x_test = x_test
		self.y_train_oh = y_train_oh
		self.y_dev_oh = y_dev_oh
		self.y_test_oh = y_test_oh

		self.drop1 = drop1
		self.drop2 = drop2
		self.drop3 = drop3
		self.drop4 = drop4
		self.drop5 = drop5
		self.drop6 = drop6
		self.drop7 = drop7
		
		self.model = Sequential()

		self.model.add(Dense(1000, activation='relu', input_shape=(1500,)))
		self.model.add(Dropout(self.drop1))

		self.model.add(Dense(700, activation='relu'))
		self.model.add(Dropout(self.drop2))

		self.model.add(Dense(500, activation='relu'))
		self.model.add(Dropout(self.drop3))

		self.model.add(Dense(500, activation='relu'))
		self.model.add(Dropout(self.drop4))

		self.model.add(Dense(300, activation='relu'))
		self.model.add(Dropout(self.drop5))

		self.model.add(Dense(300, activation='relu'))
		self.model.add(Dropout(self.drop6))

		self.model.add(Dense(256, activation='relu'))
		self.model.add(Dropout(self.drop7))

		self.model.add(Dense(256, activation='softmax'))

		self.model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')
		print("Model summary\n")
		print(self.model.summary())

	def train(self, batchSize):
		""" Train the model with the training data
		batchSize : batch size during trainig
		"""

		Epochs = 1000

		logFile = self.resultDir + '/' + self.modelName + '_' + str(batchSize) +'.log'
		csv_logger = CSVLogger(logFile, append=True, separator="\t")
		
		earlyStop = EarlyStopping(monitor='val_categorical_accuracy', patience=100, mode='auto', verbose=1, restore_best_weights=True)
		
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

		outputFile = self.resultDir + '/' + self.modelName +'_7HLw_1000_700_500_500_300_300_256_' + str(self.history.epoch[-1]+1) + 'epochs_' + 'Dropout_' + str(self.drop1).replace('.', 'p') + '_' + str(self.drop2).replace('.', 'p') + '_' + str(self.drop3).replace('.', 'p')  + '_' + str(self.drop4).replace('.', 'p')  +'_' + str(self.drop5).replace('.', 'p')  +  '_' + str(self.drop6).replace('.', 'p')  + '_' + str(self.drop7).replace('.', 'p')  + "_outputPredict.csv" 
		
		np.savetxt(outputFile, output_predict, fmt="%5.0f", delimiter=",")

		##Error Analysis of the prediction
		errorAnalysis(outputFile)

		return self.model_score

	def saveModel(self):
		""" Save the model
		"""
		saveStr = self.resultDir + '/' + self.modelName +'_7HLw_1000_700_500_500_300_300_256_' + str(self.history.epoch[-1]+1) + 'epochs_' + 'Dropout_' + str(self.drop1).replace('.', 'p') + '_' + str(self.drop2).replace('.', 'p') + '_' + str(self.drop3).replace('.', 'p')  + '_' + str(self.drop4).replace('.', 'p')  + '_' + str(self.drop5).replace('.', 'p')  + '_' + str(self.drop6).replace('.', 'p')  + '_' + str(self.drop7).replace('.', 'p')  + '_' + '{0:.2f}'.format(self.model_score[1]*100).replace('.', 'p') + '.h5'
		print("Saving model to\n%s\n" %(saveStr))
		self.model.save(saveStr)


