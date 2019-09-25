import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
#from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping
## ModelCheckPoint is written in newCallBacks
from keras.callbacks import CSVLogger, TensorBoard
import pickle
import gzip
import pandas as pd
import numpy as np
import gc
from datetime import date, datetime

from sklearn.utils import shuffle
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from keras.models import clone_model
from keras.models import load_model

# Path where new EarlyStopping and ModelCheckPoint_every_10epochs is written
import sys
sys.path.insert(0, '/extra/manojgopale/AES_data/')

import newCallBacks

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
def getData(dataPath, trainSize):
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

		## get the data for each sub part
		x_train_inter, y_train_inter = process_inputs(trainStr)
		x_dev_inter, y_dev_inter     = process_inputs(devStr)
		x_test_inter, y_test_inter   = process_inputs(testStr)

		## Substituing chunks of data to the allocated space in the array
		## The order of placement is 1,0,3,2 for the arrays
		x_train[trainSize*index*64: trainSize*(index+1)*64, : ] = x_train_inter
		x_dev[devSize*index*64: devSize*(index+1)*64, : ] = x_dev_inter
		x_test[testSize*index*64: testSize*(index+1)*64, : ] = x_test_inter

		y_train[trainSize*index*64: trainSize*(index+1)*64, 0] = y_train_inter
		y_dev[devSize*index*64: devSize*(index+1)*64, 0 ] = y_dev_inter
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
	def __init__(self, resultDir: str, modelName: str, x_train, y_train_oh, x_dev, y_dev_oh, x_test, y_test_oh, hiddenSize):
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
		self.hiddenSize = hiddenSize

		self.dropOut = 0.2
		
		self.model = Sequential()

		self.model.add(Dense(self.hiddenSize, input_shape=(1361,)))
		self.model.add(Activation('relu'))                            

		self.model.add(Dropout(self.dropOut))
		self.model.add(Dense(256))
		self.model.add(Activation('softmax'))

		self.model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')
		print("Model summary\n")
		print(self.model.summary())

	def train(self, batchSize):
		""" Train the model with the training data
		batchSize : batch size during trainig
		"""

		Epochs = 100

		logFile = self.resultDir + '/' + self.modelName +'.log'
		csv_logger = CSVLogger(logFile, append=True, separator="\t")
		
		earlyStop = newCallBacks.EarlyStopNew(monitor='val_categorical_accuracy', patience=5, mode='auto', verbose=1)
		
		filePath = self.resultDir + '/' + self.modelName + '_checkPoint_best_model.hdf5'
		## This file will include the epoch number when it gets saved.
		repeatingFile = self.resultDir + '/' + self.modelName +'_{epoch:02d}_epoch_acc_{accVal:.2f}.hdf5'
		## By default the every_10epochs will save the model at every 10 epochs
		checkPoint = newCallBacks.ModelCheckpoint_every_10epochs(filePath, repeatingFile, self.x_test, self.y_test_oh , monitor='val_categorical_accuracy', verbose=1, save_best_only=True, every_10epochs=True)
		
		self.history = self.model.fit(self.x_train, self.y_train_oh, batch_size= batchSize, epochs=Epochs, verbose=1, shuffle= True, validation_data=(self.x_dev, self.y_dev_oh), callbacks=[csv_logger, checkPoint, earlyStop])

	def evaluate(self):
		""" Evaluate the model on itself
		"""

		self.model_score = self.model.evaluate(self.x_test, self.y_test_oh, batch_size=2048)
		print("%s score = %f\n" %(self.modelName, self.model_score[1]))
		return self.model_score
	
	def saveModel(self):
		""" Save the model
		"""
		saveStr = self.resultDir + '/' + self.modelName +'_3HLw_500_500_256_noDrop_' + str(self.history.epoch[-1]+1) + 'epochs_' + str(self.dropOut).replace('.', 'p') + 'Dropout_' + '{0:.2f}'.format(self.model_score[1]*100).replace('.', 'p') + '.h5'
		print("Saving model to\n%s\n" %(saveStr))
		self.model.save(saveStr)


