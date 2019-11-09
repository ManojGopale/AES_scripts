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
## trainSize : Number of power traces per key to take
## testFlag : if true, then traina nd dev data won't be loaded
def getData(dataPath, trainSize, testFlag):
	#runDir = "/extra/manojgopale/AES_data/config3p3_15ktraining/"
	runDir = dataPath
	dataDir = runDir + "/data/"
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
		if (not testFlag):
			x_train_inter, y_train_inter = process_inputs(trainStr)
			## Substituing chunks of data to the allocated space in the array
			## The order of placement is 1,0,3,2 for the arrays
			x_train[trainSize*index*64: trainSize*(index+1)*64, : ] = x_train_inter
			y_train[trainSize*index*64: trainSize*(index+1)*64, 0] = y_train_inter
		else: 
			x_train[trainSize*index*64: trainSize*(index+1)*64, : ] = np.zeros((trainSize*64,1361))
			y_train[trainSize*index*64: trainSize*(index+1)*64, : ] = np.zeros((trainSize*64, 1))

		#if (os.stat(devStr).st_size != 0):
		if (not testFlag):
		## get the data for each sub part
			x_dev_inter, y_dev_inter     = process_inputs(devStr)
			x_dev[devSize*index*64: devSize*(index+1)*64, : ] = x_dev_inter
			y_dev[devSize*index*64: devSize*(index+1)*64, 0 ] = y_dev_inter
		else:
			x_dev[devSize*index*64: devSize*(index+1)*64, : ] = np.zeros((devSize*64, 1361))
			y_dev[devSize*index*64: devSize*(index+1)*64, : ] = np.zeros((devSize*64, 1))

		## Test data is present so check is not performed
		x_test_inter, y_test_inter   = process_inputs(testStr)
		x_test[testSize*index*64: testSize*(index+1)*64, : ] = x_test_inter
		y_test[testSize*index*64: testSize*(index+1)*64, 0 ] = y_test_inter
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
	def __init__(self, resultDir: str, modelName: str, x_train, y_train_oh, x_dev, y_dev_oh, x_test, y_test_oh, drop1, drop2, drop3):
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
		
		self.model = Sequential()

		self.model.add(Dense(500, activation='relu', input_shape=(1361,)))
		self.model.add(Dropout(self.drop1))

		self.model.add(Dense(500, activation='relu'))
		self.model.add(Dropout(self.drop2))

		self.model.add(Dense(256, activation='relu'))
		self.model.add(Dropout(self.drop3))

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

		self.model_score = self.model.evaluate(self.x_test, self.y_test_oh, batch_size=2048)
		print("%s score = %f\n" %(self.modelName, self.model_score[1]))

		##Saving atucal vs predicted predictions
		##np.argmax returns the index where it see's 1 in the row
		y_pred = np.argmax(self.model.predict(self.x_test, batch_size=2048), axis=1)

		## vstack will stack them in 2 rows, so we use Trasnpose to get them in column stack
		output_predict = np.vstack((np.argmax(self.y_test_oh, axis=1), y_pred)).T
		outputFile = self.resultDir + "/outputPredict.csv"  
		np.savetxt(outputFile, output_predict, fmt="%5.0f", delimiter=",")

		##Error Analysis of the prediction
		errorAnalysis(outputFile)

		return self.model_score
	
	def saveModel(self):
		""" Save the model
		"""
		saveStr = self.resultDir + '/' + self.modelName +'_3HLw_500_500_256_' + str(self.history.epoch[-1]+1) + 'epochs_' + 'Dropout_' + str(self.drop1).replace('.', 'p') + '_' + str(self.drop2).replace('.', 'p') + '_' + str(self.drop3).replace('.', 'p') + '_' + '{0:.2f}'.format(self.model_score[1]*100).replace('.', 'p') + '.h5'
		print("Saving model to\n%s\n" %(saveStr))
		self.model.save(saveStr)


