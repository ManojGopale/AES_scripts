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
from datetime import date, datetime

from sklearn.utils import shuffle
import sklearn, random
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from keras.models import clone_model
from keras.models import load_model

import scipy.io as si
import argparse
import time

## Dir 
runDir = "/xdisk/manojgopale/AES/dataCollection/"

np.random.seed(9)
scaler = StandardScaler()
def process_inputs (data):
	#data = pd.read_csv(dataPath, header=None)
	dataShuffle = shuffle(data)
	x_data_shuffle = dataShuffle[:,0:-1]
	y_data = dataShuffle[:,-1]
	x_data = scaler.fit_transform(x_data_shuffle)
	return x_data, y_data

## config: the configuration for which data is to be generated 
## trainSize (4000) : Number of power traces per key to take
def getData(config, trainSize):

	## runDir
	runDir = "/xdisk/manojgopale/AES/dataCollection/"

	## Count for saving files
	count = 0
	
	## Number of keys to save in one train file
	partLen = 128
	keySize = 256

	## Create output 
	inter_y = np.zeros([trainSize,1])
	inter_y[1999] = 1 ## Settting the 2000th data to be '1', which is the the AES trace

	for key in range(keySize):
		## Get 3 random samples for adding train, dev and test to the data
		index = random.sample(range(500), 3)

		## Load each key data for creating the training and testing files
		matStr = runDir + "/matResult/" +  config + "/value" + str(key) + ".mat"
		print("key=%s\nmatStr=%s\n" %(key, matStr))

		## Create intermediate datasets which will be post processed to get data 
		## split into 1361 power traces
		interTrain = si.loadmat(matStr)["power"][index[0]]
		interDev =   si.loadmat(matStr)["power"][index[1]]	
		interTest =  si.loadmat(matStr)["power"][index[2]]

		## Create data sets, this splits the complete data in 4000 rows with 1361 in each row
		## Each is a wondow of 1361, with step of 1
		interTrainData = interTrain[np.array([range(i, i+1361) for i in range(trainSize)])]
		interDevData =   interDev[np.array([range(i, i+1361) for i in range(trainSize)])]
		interTestData =  interTest[np.array([range(i, i+1361) for i in range(trainSize)])]

		##Append the output's at 2000th location for correct traces
		interTrainData = np.concatenate((interTrainData, inter_y), axis=1)
		interDevData =   np.concatenate((interDevData, inter_y), axis=1)
		interTestData =  np.concatenate((interTestData, inter_y), axis=1)

		## Append inter*Data to coressponding full data.
		if key == 0 or key%partLen == 0:
			fullTrain = interTrainData[:]
			fullDev = interDevData[:]
			fullTest = interTestData[:]
			print("loop1: key=%s" %(key))

		elif key%partLen == partLen-1:
			fullTrain = np.concatenate((fullTrain, interTrainData), axis=0)
			fullDev =   np.concatenate((fullDev, interDevData), axis=0)
			fullTest =  np.concatenate((fullTest, interTestData), axis=0)

			## Save files to csv, so as to use it for portability analysis
			trainSave = runDir + "/processedData/run_1_per_key/data/" + config + "/train_" + str(count) + ".csv"
			devSave   = runDir + "/processedData/run_1_per_key/data/" + config + "/dev_" + str(count) + ".csv"
			testSave  = runDir + "/processedData/run_1_per_key/data/" + config + "/test_" + str(count) + ".csv"

			##Save files
			np.savetxt(trainSave, fullTrain, fmt="%10.5f", delimiter=",")
			np.savetxt(devSave, fullTrain, fmt="%10.5f", delimiter=",")
			np.savetxt(testSave, fullTrain, fmt="%10.5f", delimiter=",")

			## Increment the count variable
			count = count + 1

			print("loop2: key=%s" %(key))
			print("\nSaved data to\n%s\n%s\n%s" %(trainSave, devSave, testSave))
			
		else:
			fullTrain = np.concatenate((fullTrain, interTrainData), axis=0)
			fullDev =   np.concatenate((fullDev, interDevData), axis=0)
			fullTest =  np.concatenate((fullTest, interTestData), axis=0)
			print("loop3: key=%s" %(key))
			

		## Clear variables
		interTrain = None
		interDev = None
		interTest = None
		interTrainData = None
		interDevData = None
		interTestData = None

	##Create final datasets
	x_train, y_train = process_inputs(fullTrain)
	x_dev, y_dev = process_inputs(fullDev)
	x_test, y_test = process_inputs(fullTest)

	##Clear fullData variables
	fullTrain = None
	fullDev = None
	fullTest = None

	print("\nCleared fullData variables\n")

	print("Shapes of datasets\nx_train=%s, y_train=%s\nx_dev=%s, y_dev=%s\nx_test=%s, y_test=%s" 
	       %(x_train.shape, y_train.shape, x_dev.shape, y_dev.shape, x_test.shape, y_test.shape))

	return (x_train, y_train), (x_dev, y_dev), (x_test, y_test)

class Classifier:
	def __init__(self, resultDir: str, modelName: str, x_train, y_train, x_dev, y_dev, x_test, y_test, hiddenSize):
		""" Initialize parameters and sequential model for training
		"""
		self.resultDir = resultDir
		self.modelName = modelName
		self.x_train = x_train
		self.x_dev = x_dev
		self.x_test = x_test
		self.y_train = y_train
		self.y_dev = y_dev
		self.y_test = y_test
		self.hiddenSize = hiddenSize

		self.dropOut = 0.2
		
		self.model = Sequential()

		self.model.add(Dense(self.hiddenSize, input_shape=(1361,)))
		self.model.add(Activation('relu'))                            

		self.model.add(Dropout(self.dropOut))
		self.model.add(Dense(1))
		self.model.add(Activation('sigmoid'))

		self.model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer='adam')
		print("Model summary\n")
		print(self.model.summary())

	def train(self, batchSize):
		""" Train the model with the training data
		batchSize : batch size during trainig
		"""

		Epochs = 100

		logFile = self.resultDir + '/' + self.modelName +'.log'
		csv_logger = CSVLogger(logFile, append=True, separator="\t")
		
		earlyStop = EarlyStopping(monitor='val_binary_accuracy', patience=5, mode='auto', verbose=1, restore_best_weights=True)
		
		#filePath = self.resultDir + '/' + self.modelName + '_checkPoint_best_model.hdf5'
		### This file will include the epoch number when it gets saved.
		#repeatingFile = self.resultDir + '/' + self.modelName +'_{epoch:02d}_epoch_acc_{accVal:.2f}.hdf5'
		### By default the every_10epochs will save the model at every 10 epochs
		#checkPoint = newCallBacks.ModelCheckpoint_every_10epochs(filePath, repeatingFile, self.x_test, self.y_test , monitor='val_categorical_accuracy', verbose=1, save_best_only=True, every_10epochs=True)

		##Class weights for class imbalanced data
		## https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
		y = self.y_train.astype(int)

		sampleSize = len(y)
		classNum = len(set(y))

		weight_0 = sampleSize/(classNum * np.bincount(y)[0])
		weight_1 = sampleSize/(classNum * np.bincount(y)[1])

		classWeight = {0: weight_0, 1: weight_1}
		
		self.history = self.model.fit(self.x_train, self.y_train, batch_size= batchSize, epochs=Epochs, verbose=1, shuffle= True, validation_data=(self.x_dev, self.y_dev), class_weight=classWeight, callbacks=[csv_logger, earlyStop])

	def evaluate(self):
		""" Evaluate the model on itself
		"""

		self.model_score = self.model.evaluate(self.x_test, self.y_test, batch_size=2048)
		print("%s score = %f\n" %(self.modelName, self.model_score[1]))
		return self.model_score
	
	def saveModel(self):
		""" Save the model
		"""
		saveStr = self.resultDir + '/' + self.modelName +'1HL_' + str(self.hiddenSize) + '_' + str(self.history.epoch[-1]+1) + 'epochs_' + str(self.dropOut).replace('.', 'p') + 'Dropout_' + '{0:.2f}'.format(self.model_score[1]*100).replace('.', 'p') + '.h5'
		print("Saving model to\n%s\n" %(saveStr))
		self.model.save(saveStr)


