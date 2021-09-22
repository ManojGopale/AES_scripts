import pandas as pd
import numpy as np
import os
import copy
from sklearn.utils import shuffle
from keras.utils import np_utils

class Data:
	def processInput(self, dataPath, tracesPerKey):
		data = pd.read_csv(dataPath, header=None).sample(frac=1).reset_index(drop=True)##Added frac=1, shuffle on 7/15/21
		## Create a loop that takes fixed amount of data while generating the traces
		interData=pd.DataFrame()
		## get unique keys in the dataset
		for key in pd.unique(data.iloc[:,-1]):
			## Filter out traces per key and only take tracesPerKey of traces per key
			interData=pd.concat([interData,data.loc[data.iloc[:,-1]==key].iloc[0:tracesPerKey,:]])

		## https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
		## sample with frac=1, reshuffles the entire data and returns it back
		interData = interData.sample(frac=1).reset_index(drop=True)
		#Mx_data_shuffle = dataShuffle.iloc[:,0:-1]
		#My_data = dataShuffle.iloc[:,-1]
		##NOTE: Comment last line after this experiment
		#x_data = scaler.fit_transform(x_data_shuffle)
		return interData.iloc[:,0:-1], interData.iloc[:,-1]

	## dataSetType -> "Train", "Dev", "Test"
	def getData(self, dataPath, configName, tracesPerKey, dataSetType):
		x_data = pd.DataFrame()
		y_data = pd.DataFrame()
		for index, val in enumerate(range(8)):
			csvPath= dataPath + "/" + configName + "/aesData_config9_" + dataSetType + "_" + str(val) + ".csv" 
			if(os.path.isfile(csvPath)):
				print("Loading data from %s\n" %(csvPath))
				xInter, yInter = self.processInput(csvPath, tracesPerKey)
				## ignore_index=True, restets the index and does not carry forward the original index's of datasets
				x_data = pd.concat([x_data, xInter], ignore_index=True)
				y_data = pd.concat([y_data, yInter], ignore_index=True)
			else:
				print("\nFile= %s, does not exists\n" %(csvPath))
		return x_data, y_data

	def shuffleData(self, x_data, y_data):
		"""Shuffle the data, with random_state= None as default
		"""
		##NOTE: If there is any problem's with shuffling, go back to the utlis.shuffle function
		## Shuffling
		## https://scikit-learn.org/stable/modules/generated/sklearn.utils.shuffle.html
		#Mx_data, y_data = shuffle(x_data, y_data, random_state=None)
		print("\nStarted shuffling of data\nx_data[0]= %s\ny_data[0]= %s" %(x_data.iloc[0], y_data.iloc[0]))
		print(*x_data.iloc[0])
		print("\nx_data[100]= %s\ny_data[100]= %s" %(x_data.iloc[100], y_data.iloc[100]))
		print(*x_data.iloc[100])
		newIndex = np.random.permutation(x_data.index)
		## incase the index's from concat are duplicate, we will reset them before shuffling
		x_data.reset_index(drop=True, inplace=True)
		y_data.reset_index(drop=True, inplace=True)

		## the idea here is to shuffle the index keeping the order same and then reshuffling them to get the shuffled df
		## set_index sets the index's to the newIndex without changing the order.
		## we will have to sort them, so that the new order is randomized
		x_data.set_index(newIndex, inplace=True)
		y_data.set_index(newIndex, inplace=True)

		## Sorting indexes based on the index we just set
		x_data.sort_index(inplace=True)
		y_data.sort_index(inplace=True)

		print("\nFinished shuffling of data\nx_data[0]= %s\ny_data[0]= %s" %(x_data.iloc[0], y_data.iloc[0]))
		print(*x_data.iloc[0])
		print("\nx_data[100]= %s\ny_data[100]= %s" %(x_data.iloc[100], y_data.iloc[100]))
		print(*x_data.iloc[100])
		return x_data, y_data
	
	def oneHotY(self, y_data):
		"""One hot the y_values in the dataset
		"""
		## One hot assignment
		n_classes = 256
		y_data_oh = np_utils.to_categorical(y_data, n_classes)
		
		print("\nOne-hot encoded for outputs\n")
		return y_data_oh
		
	def getStdParam(self, x_data):
		"""Get mean and std deviation of the x_data
		"""
		## get mean and std dev. for each column with axis=0
		x_data_mean = x_data.mean(axis=0)
		x_data_std  = x_data.std(axis=0)
		return x_data_mean, x_data_std
	
	def stdData(self, x_data, meanToApply, stdToApply):
		""" Standardize the x_data, by the mean and std deviation 
		Make sure the x_data is in numpy format
		"""
		##Mprint("Started normalizing\n")
		##M## since we are having mean and std dev. for each column, we will have to sample each row to apply them, hence axis=1
		##Mreturn x_data.apply(lambda x: (x-meanToApply)/stdToApply, axis=1).to_numpy()
		print("Started normalizing\n")
		chunkSize = 28000
		if(x_data.shape[0]>chunkSize):
			chunkNum = int(x_data.shape[0]/chunkSize)
			for chunkIndex in range(chunkNum):
				print("chunkIndx= %s, chunkNum = %s" %(chunkIndex, chunkNum))
				if(chunkIndex != chunkNum-1): 
					x_data[chunkIndex*chunkSize: (chunkIndex+1)*chunkSize] = (x_data[chunkIndex*chunkSize: (chunkIndex+1)*chunkSize]-meanToApply)/stdToApply
				else:
					x_data[chunkIndex*chunkSize: ] = (x_data[chunkIndex*chunkSize: ] - meanToApply)/stdToApply
		else:
			print("ChunkSize less than 28000\n")
			x_data[0: ] = (x_data[0: ] - meanToApply)/stdToApply

		return x_data
			
		### Row std
		## x_train = x_train.apply(lambda x: (x-x.mean())/x.std(), axis=1)
	def stdDataRowWise(self, x_data):
		"""	Apply mean and standard dev to the row instead of column.
				Make usre the x_data is a numpy.ndarray
		"""
		print("Started row wise standardizing\n")
		chunkSize = 28000
		if(x_data.shape[0]>chunkSize):
			chunkNum = int(x_data.shape[0]/chunkSize)
			for chunkIndex in range(chunkNum):
				print("chunkIndx= %s, chunkNum = %s" %(chunkIndex, chunkNum))
				if(chunkIndex != chunkNum-1): 
					meanToApply = copy.deepcopy(x_data[chunkIndex*chunkSize: (chunkIndex+1)*chunkSize].mean(axis=1))
					meanToApply = meanToApply.reshape(meanToApply.shape[0],1)
					stdToApply = copy.deepcopy(x_data[chunkIndex*chunkSize: (chunkIndex+1)*chunkSize].std(axis=1))
					stdToApply = stdToApply.reshape(stdToApply.shape[0],1)
					x_data[chunkIndex*chunkSize: (chunkIndex+1)*chunkSize] = (x_data[chunkIndex*chunkSize: (chunkIndex+1)*chunkSize]-meanToApply)/stdToApply
				else:
					meanToApply = copy.deepcopy(x_data[chunkIndex*chunkSize: ].mean(axis=1))
					meanToApply = meanToApply.reshape(meanToApply.shape[0],1)
					stdToApply = copy.deepcopy(x_data[chunkIndex*chunkSize: ].std(axis=1))
					stdToApply = stdToApply.reshape(stdToApply.shape[0],1)
					x_data[chunkIndex*chunkSize: ] = (x_data[chunkIndex*chunkSize: ] - meanToApply)/stdToApply
		else:
			print("ChunkSize less than 28000\n")
			meanToApply = copy.deepcopy(x_data[0: ].mean(axis=1))
			meanToApply = meanToApply.reshape(meanToApply.shape[0],1)
			stdToApply = copy.deepcopy(x_data[0: ].std(axis=1))
			stdToApply = stdToApply.reshape(stdToApply.shape[0],1)
			x_data[0: ] = (x_data[0: ] - meanToApply)/stdToApply

		return x_data

