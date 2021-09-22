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
	x_data = scaler.fit_transform(data.trace.apply(pd.Series).values)

	## Without transformation
	##MG x_data = data.trace.apply(pd.Series).values
	return x_data, y_data

## dataPath : Path where the working directory is located. 
## trainSize : Number of power traces per key to take
## testFlag : if true, then traina nd dev data won't be loaded
def getData(dataPath, moreDataPath, trainSize, trainFlag, devFlag, testFlag):
	##TODO changed devSize to 1000 for trails for more trainSize data. It was 5000 for default run
	devSize  = 1000
	testSize = 1000
	numTraces = 1500 ##Number of traces collected per key
	otherKeyNum = 100

	## Pre defining the arrays based on sizes of the data
	x_train = np.zeros(((28000+255*otherKeyNum), numTraces))
	x_dev = np.zeros((devSize*256, numTraces))
	x_test = np.zeros((testSize*256, numTraces))

	y_train = np.zeros(((28000+255*otherKeyNum), 1))
	y_dev = np.zeros((devSize*256, 1))
	y_test = np.zeros((testSize*256, 1))

	for index, val in enumerate(range(0,256)):
		print("Started data processing for %d key\n" %(val))
		trainStr = dataPath + "train_" + str(val) + ".pkl.zip"
		devStr   = dataPath + "dev_" + str(val) + ".pkl.zip"
		testStr  = dataPath + "test_" + str(val) + ".pkl.zip"

		##more training data path
		moreTrainStr = moreDataPath + "train_" + str(val) + ".pkl.zip"

		## Checking if the file size is 0, before processing data
		## This check is for cross config analysis, where traina nd dev are empty
		#if (os.stat(trainStr).st_size != 0):
		if (trainFlag):
			if(val==205):
				x_train_inter, y_train_inter = process_inputs(trainStr)
				## Trainsize will still be 15000, but we will take data from devSet to trainset
				x_train[otherKeyNum*index: otherKeyNum*(index) + 15000, : ] = x_train_inter
				y_train[otherKeyNum*index: otherKeyNum*(index) + 15000, : ] = np.ones((15000,1))

				## Adding 9000 more data
				x_train_inter_more, y_train_inter_more = process_inputs(moreTrainStr)
				x_train[otherKeyNum*(index) + 15000: (otherKeyNum*(index) + 15000) + 9000, : ] = x_train_inter_more[0:9000, :]
				y_train[otherKeyNum*(index) + 15000: (otherKeyNum*(index) + 15000) + 9000, : ] = np.ones((9000,1))

				print("Train= %s\n" %(trainFlag))
			elif(val<205):
				x_train_inter, y_train_inter = process_inputs(trainStr)
				## Trainsize will still be 15000, but we will take data from devSet to trainset
				x_train[otherKeyNum*index: otherKeyNum*(index+1), : ] = x_train_inter[0:otherKeyNum, :]
				y_train[otherKeyNum*index: otherKeyNum*(index+1), : ] = np.zeros((otherKeyNum,1))

				print("Train= %s\n" %(trainFlag))
			else:
				x_train_inter, y_train_inter = process_inputs(trainStr)
				## Trainsize will still be 15000, but we will take data from devSet to trainset
				x_train[otherKeyNum*(index-1)+28000: otherKeyNum*(index-1)+28000+otherKeyNum, : ] = x_train_inter[0:otherKeyNum, :]
				y_train[otherKeyNum*(index-1)+28000: otherKeyNum*(index-1)+28000+otherKeyNum, : ] = np.zeros((otherKeyNum,1))

				print("Train= %s\n" %(trainFlag))

		else: 
			## Assigning the array's to 0's
			##NOTE: needs to change shape, but since we are always training, I am not changing this
			x_train[trainSize*index: trainSize*(index+1), : ] = np.zeros((trainSize,numTraces))
			y_train[trainSize*index: trainSize*(index+1), : ] = np.zeros((trainSize, 1))
			print("train= %s\n" %(trainFlag))

		#if (os.stat(devStr).st_size != 0):
		if (devFlag):
		## get the data for each sub part
			if(val==205):
				x_dev_inter, y_dev_inter     = process_inputs(devStr)
				print("x_dev_inter= %s, y_dev_inter= %s" %(x_dev_inter.shape, y_dev_inter.shape))
				x_dev[devSize*index: devSize*(index+1), : ] = x_dev_inter[0:devSize, :]
				y_dev[devSize*index: devSize*(index+1), : ] = np.ones((devSize, 1))
				print("Dev= %s\n" %(devFlag))
				print("x_dev= %s, y_dev= %s" %(x_dev.shape, y_dev.shape))

				#M## Adding 4000 traces to trainSet here
				#Mx_train[otherKeyNum*(index) + 15000 +9000: (otherKeyNum*(index) + 15000) + 13000, : ] = x_dev_inter[devSize:5000, :]
				#My_train[otherKeyNum*(index) + 15000 +9000: (otherKeyNum*(index) + 15000) + 13000, : ] = np.ones((4000,1))
				#Mprint("x_trainSize = %s, y_trainSize= %s" %(x_train.shape, y_train.shape))
			else:
				x_dev_inter, y_dev_inter     = process_inputs(devStr)
				print("x_dev_inter= %s, y_dev_inter= %s" %(x_dev_inter.shape, y_dev_inter.shape))
				x_dev[devSize*index: devSize*(index+1), : ] = x_dev_inter[0:devSize, :]
				y_dev[devSize*index: devSize*(index+1), : ] = np.zeros((devSize, 1))
				print("Dev= %s\n" %(devFlag))
				print("x_dev= %s, y_dev= %s" %(x_dev.shape, y_dev.shape))

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
			print("x_test= %s, y_test= %s" %(x_test.shape, y_test.shape))
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
	x_train_inter_more = None
	y_train_inter_more = None
	print("\nCleared variables\n")

	##Not shuffling for debugging, should be removed
	## Shuffling
	## https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html
	print("\nStarted shuffling of data\nx_train[0]= %s\ny_train[0]= %s" %(x_train[0], y_train[0]))
	print("\nx_train[12000]= %s\ny_train[12000]= %s" %(x_train[12000], y_train[12000]))
	x_train, y_train = shuffle(x_train, y_train, random_state=0)
	x_dev, y_dev = shuffle(x_dev, y_dev, random_state=0)
	x_test, y_test = shuffle(x_test, y_test, random_state=0)
	print("\nFinished shuffling of data\nx_train[0]= %s\ny_train[0]= %s" %(x_train[0], y_train[0]))
	print("\nx_train[12000]= %s\ny_train[12000]= %s" %(x_train[12000], y_train[12000]))
	print("x_dev stats: %s\ny_dev stats= %s\n" %(np.unique(x_dev, return_counts=True), np.unique(y_dev, return_counts=True)))

	##NOTE: Remove:
	#Mimport pdb; pdb.set_trace()
	## One hot assignment
	#Mn_classes = 256
	#My_train = np_utils.to_categorical(y_train, n_classes)
	#My_dev = np_utils.to_categorical(y_dev, n_classes)
	#My_test = np_utils.to_categorical(y_test, n_classes)
	
	print("\nOne-hot encoded for outputs\n")

	return (x_train, y_train), (x_dev, y_dev), (x_test, y_test)

class Classifier:
	def __init__(self, resultDir: str, modelName: str, x_train, y_train, x_dev, y_dev, x_test, y_test, drop1, drop2, drop3):
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

		self.drop1 = drop1
		self.drop2 = drop2
		self.drop3 = drop3
		
		self.model = Sequential()

		self.model.add(Dense(1500, activation='relu', input_shape=(1500,)))
		self.model.add(Dropout(self.drop1))

		self.model.add(Dense(700, activation='relu'))
		self.model.add(Dropout(self.drop2))

		self.model.add(Dense(700, activation='relu'))
		self.model.add(Dropout(self.drop3))

		self.model.add(Dense(1000, activation='relu'))
		self.model.add(Dropout(self.drop3))

		self.model.add(Dense(1, activation='sigmoid'))

		self.model.compile(loss='binary_crossentropy', metrics=['binary_accuracy'], optimizer='adam')
		print("Model summary\n")
		print(self.model.summary())

	def train(self, batchSize):
		""" Train the model with the training data
		batchSize : batch size during trainig
		"""

		Epochs = 100

		logFile = self.resultDir + '/' + self.modelName + '_' + str(batchSize) +'.log'
		csv_logger = CSVLogger(logFile, append=True, separator="\t")
		
		earlyStop = EarlyStopping(monitor='val_binary_accuracy', patience=10, mode='auto', verbose=1, restore_best_weights=True)
		
		##filePath = self.resultDir + '/' + self.modelName + '_checkPoint_best_model.hdf5'
		#### This file will include the epoch number when it gets saved.
		##repeatingFile = self.resultDir + '/' + self.modelName +'_{epoch:02d}_epoch_acc_{accVar:.2f}.hdf5'
		#### By default the every_10epochs will save the model at every 10 epochs
		##checkPoint = newCallBacks.ModelCheckpoint_every_10epochs(filePath, repeatingFile, self.x_test, self.y_test , monitor='val_categorical_accuracy', verbose=1, save_best_only=True, every_10epochs=True)
		
		self.history = self.model.fit(self.x_train, self.y_train, batch_size= batchSize, epochs=Epochs, verbose=1, shuffle= True, validation_data=(self.x_dev, self.y_dev), callbacks=[csv_logger, earlyStop])

	def evaluate(self):
		""" Evaluate the model on itself
		"""

		## We should be evaluating on dev dataset as well, so commenting x_test
		#self.model_score = self.model.evaluate(self.x_test, self.y_test, batch_size=2048)
		self.model_score = self.model.evaluate(self.x_dev, self.y_dev, batch_size=2048)
		print("%s score, accu  = %f\n" %(self.modelName, self.model_score[1]))

		##Saving atucal vs predicted predictions
		##np.argmax returns the index where it see's 1 in the row
		#y_pred = np.argmax(self.model.predict(self.x_test, batch_size=2048), axis=1)
		y_pred = self.model.predict(self.x_dev, batch_size=2048)

		## vstack will stack them in 2 rows, so we use Trasnpose to get them in column stack
		#output_predict = np.vstack((np.argmax(self.y_test, axis=1), y_pred)).T
		output_predict = np.vstack((self.y_dev, y_pred)).T

		outputFile = self.resultDir + '/' + self.modelName +'_3HLw_1000_700_500_' + str(self.history.epoch[-1]+1) + 'epochs_' + 'Dropout_' + str(self.drop1).replace('.', 'p') + '_' + str(self.drop2).replace('.', 'p') + '_' + str(self.drop3).replace('.', 'p')  + '_' +  "_outputPredict.csv" 
		
		np.savetxt(outputFile, output_predict, fmt="%5.0f", delimiter=",")

		##Error Analysis of the prediction
		errorAnalysis(outputFile)

		return self.model_score

	def saveModel(self):
		""" Save the model
		"""
		saveStr = self.resultDir + '/' + self.modelName +'_3HLw_1000_700_500_' + str(self.history.epoch[-1]+1) + 'epochs_' + 'Dropout_' + str(self.drop1).replace('.', 'p') + '_' + str(self.drop2).replace('.', 'p') + '_' + str(self.drop3).replace('.', 'p')  + '_' + '{0:.2f}'.format(self.model_score[1]*100).replace('.', 'p') + '.h5'
		print("Saving model to\n%s\n" %(saveStr))
		self.model.save(saveStr)


