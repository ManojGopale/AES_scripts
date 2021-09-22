import sys
sys.path.insert(0, '/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/')
import classify_general
import time
import numpy as np
import pandas as pd
import os

import scipy.io as si
import numpy as np
import argparse
import pickle
import gzip
import time
from optparse import OptionParser
import random

########
## csvConfig -> The config for which the csv file is loaded
## trainPartitionNum -> The partition number (0-7) of the train file which is to be used for this run
## startKey -> start of the key ranges that is used for this run
## endKey -> end of the key ranges that is used for this run
########

parser = OptionParser()
parser.add_option('--csvConfig',
									action = 'store', type='string', dest='csvConfig', default = 'config3p1')
parser.add_option('--trainPartitionNum',
									action = 'store', type='int', dest='trainPartitionNum', default = 0)
parser.add_option('--startKey',
									action = 'store', type='int', dest='startKey', default = 0)
parser.add_option('--endKey',
									action = 'store', type='int', dest='endKey', default = 32)

(options, args) = parser.parse_args()
########
csvConfig = options.csvConfig
trainPartitionNum = options.trainPartitionNum
startKey = options.startKey
endKey = options.endKey

########
## We can directly load the csv files,
csvFilePath="/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/" + csvConfig + "/aesData_config9_Train_" + str(trainPartitionNum) + ".csv"
print("Csv file read= %s\n" %(csvFilePath))
train=pd.read_csv(csvFilePath, header=None)

parentDir = "/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/"
## this is the matlab config to be compared against the csv data of 4.1 and 3.3
#Mcsv_matConfig = "config3p1"
## this is the matlab data to be compared against csv_mat config which we used earlier
#Mmat_matConfig = "config5p1"

## List of matlab config's that we can use for comaprison
## We will select 5 configs for csv comapre and 5 more for matlab compares
#MconfigCompare=["config3p1", "config3p2", "config3p3","config3p4", "config4p1", "config4p2", "config4p3","config4p4", "config5p1", "config5p2","config5p3", "config5p4"]
configCompare=["config4p2", "config4p3", "config4p4", "config5p2", "config3p2"]

## Generate 10 random keys in between 0-31 inclusive, for our trails
for key in random.sample(range(startKey, endKey), 32):
	#Mkey = 11 ## Can change the key too
	print("key= %s\n" %(key))
	## Get the data for a specific key
	train_key = train.loc[train.iloc[:,1500] == key].iloc[:,:-1]
	
	for csv_matConfig in random.sample(configCompare, 5):
		## Compare csv's data too
		keyStrCsv = "/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/" + csv_matConfig + "/aesData_config9_Train_" + str(trainPartitionNum) + ".csv"
		print ("csv data for csv-csv comparison is %s" %(keyStrCsv))
		## Check if the file exists
		if(os.path.isfile(keyStrCsv)):
			try:
				train_forCsv=pd.read_csv(keyStrCsv, header=None)
			except:
				print("Exception occured while loading csv file for %s, error= %s\n" %(keyStrCsv, sys.exc_info()))
			else:
				train_key_forCsv = train_forCsv.loc[train_forCsv.iloc[:,1500] == key].iloc[:,:-1]
				csv_csv_compare = pd.concat([train_key,train_key_forCsv])
				print("csv1 data of %s, csv2 data of %s, key= %s\n" %(csvConfig,csv_matConfig, key))
				print("Shape before finding duplicates = %s, After dropping duplicates= %s\n" %(csv_csv_compare.shape,csv_csv_compare.drop_duplicates().shape))
				if(csv_csv_compare.shape[0]/2 == csv_csv_compare.drop_duplicates().shape[0]):
					if(csvConfig == csv_matConfig):
						print("-C- Csv of %s and csv of %s have identical data for key= %s\n" %(csvConfig, csv_matConfig, key))
					else:
						print("*C* Csv of %s and csv of %s have identical data for key= %s\n" %(csvConfig, csv_matConfig, key))
				elif(csv_csv_compare.shape != csv_csv_compare.drop_duplicates().shape):
					if(csvConfig == csv_matConfig):
						print("-C- Csv of %s %s and csv of %s %s have %s common data for key= %s\n" %(csvConfig, train_key.shape, csv_matConfig, train_key_forCsv.shape ,csv_csv_compare.shape[0]-csv_csv_compare.drop_duplicates().shape[0] ,key))
					else:
						print("#C# Csv of %s %s and csv of %s %s have %s common data for key= %s\n" %(csvConfig, train_key.shape, csv_matConfig, train_key_forCsv.shape ,csv_csv_compare.shape[0]-csv_csv_compare.drop_duplicates().shape[0] ,key))
		else:
			print("%s does not exixts\n" %(keyStrCsv))

				
		## Matlab data for comparing against csv data
		keyStr_forCsv = parentDir  + '/matResult/' + csv_matConfig + '/value' + str(key) + '.mat'
		print("matlab data for csv comparison is taken from\n%s\n" %(keyStr_forCsv))
		## Check if the file exists before comapring and loading
		if (os.path.isfile(keyStr_forCsv)):
			try:
				keyData_forCsv = si.loadmat(keyStr_forCsv)
			except:
				print("Exception occured while loadMat for %s, error= %s\n" %(keyStr_forCsv, sys.exc_info()))
			else:
				keyDataArray_forCsv = keyData_forCsv['power'].reshape(keyData_forCsv['power'].shape)[:,0:1500]
				
				csv_mat_key = pd.DataFrame(data=keyDataArray_forCsv)
				## Compare csv vs matlab
				csv_mat_compare = pd.concat([csv_mat_key,train_key])
				print("csv data of %s %s, mat data of %s %s, key= %s\n" %(csvConfig, train_key.shape, csv_matConfig, csv_mat_key.shape ,key))
				print("Shape before finding duplicates = %s, After dropping duplicates= %s\n" %(csv_mat_compare.shape,csv_mat_compare.drop_duplicates().shape))
				if(csv_mat_compare.shape[0]/2 == csv_mat_compare.drop_duplicates().shape[0]):
					if(csvConfig == csv_matConfig):
						print("-C- Csv of %s and matlab of %s have identical data for key= %s\n" %(csvConfig, csv_matConfig, key))
					else:
						print("*C* Csv of %s and matlab of %s have identical data for key= %s\n" %(csvConfig, csv_matConfig, key))
				elif(csv_mat_compare.shape != csv_mat_compare.drop_duplicates().shape):
					if(csvConfig == csv_matConfig):
						print("-C- Csv of %s %s and matlab of %s %s have %s common data for key= %s\n" %(csvConfig, train_key.shape, csv_matConfig, csv_mat_key.shape ,csv_mat_compare.shape[0]-csv_mat_compare.drop_duplicates().shape[0] ,key))
					else:
						print("#C# Csv of %s %s and matlab of %s %s have %s common data for key= %s\n" %(csvConfig, train_key.shape, csv_matConfig, csv_mat_key.shape ,csv_mat_compare.shape[0]-csv_mat_compare.drop_duplicates().shape[0] ,key))
		else:
			print("%s does not exixts\n" %(keyStr_forCsv))

		
		for mat_matConfig in random.sample(configCompare, 5):
			## Generate another config matlab data to comapre against the csv_mat key
			keyStr_forMat = parentDir  + '/matResult/' + mat_matConfig + '/value' + str(key) + '.mat'
			## Check if the file exixts
			if(os.path.isfile(keyStr_forMat) and os.path.isfile(keyStr_forCsv)):
				print("Matlab data for mat_mat comparison is taken from\n%s\n" %(keyStr_forMat))
				try:
					keyData_forMat = si.loadmat(keyStr_forMat)
					keyData_forCsv = si.loadmat(keyStr_forCsv)
				except:
					print("Exception occured while loadMat for %s or %s, error= %s\n" %(keyStr_forMat, keyStr_forCsv, sys.exc_info()))
				else:
					keyDataArray_forMat = keyData_forMat['power'].reshape(keyData_forMat['power'].shape)[:,0:1500]
					
					mat_mat_key = pd.DataFrame(data=keyDataArray_forMat)
					mat_mat_compare = pd.concat([mat_mat_key, csv_mat_key])
					
					print("Mat1 data of %s %s Mat2 data of %s %s, key= %s\n" %(csv_matConfig, csv_mat_key.shape, mat_matConfig, mat_mat_key.shape ,key))
					print("Shape before finding duplicates = %s, After dropping duplicates= %s\n" %(mat_mat_compare.shape,mat_mat_compare.drop_duplicates().shape))
					if(mat_mat_compare.shape[0]/2 == mat_mat_compare.drop_duplicates().shape[0]):
						if(csv_matConfig == mat_matConfig):
							print("-M- Matlab of %s and matlab of %s have identical data for key = %s\n" %(csv_matConfig, mat_matConfig, key))
						else:
							print("*M* Matlab of %s and matlab of %s have identical data for key = %s\n" %(csv_matConfig, mat_matConfig, key))
					elif(mat_mat_compare.shape != mat_mat_compare.drop_duplicates().shape):
						if(csv_matConfig == mat_matConfig):
							print("-M- Matlab of %s %s and matlab of %s %s have %s common data for key = %s\n" %(csv_matConfig, csv_mat_key.shape ,mat_matConfig, mat_mat_key.shape, mat_mat_compare.shape[0]-mat_mat_compare.drop_duplicates().shape[0] ,key))
						else:
							print("#M# Matlab of %s %s and matlab of %s %s have %s common data for key = %s\n" %(csv_matConfig, csv_mat_key.shape ,mat_matConfig, mat_mat_key.shape, mat_mat_compare.shape[0]-mat_mat_compare.drop_duplicates().shape[0] ,key))
			elif(not os.path.isfile(keyStr_forMat)):
				print("%s does not exists\n" %(keyStr_forMat))

			## Reverting the variables to none
			keyStr_forMat = None
			keyData_forMat = None
			keyDataArray_forMat = None
			mat_mat_key = None
			mat_mat_compare = None

		## Reverting varibles from this loop
		keyStr_forCsv = None
		keyData_forCsv = None
		keyDataArray_forCsv = None
		csv_mat_key = None
		csv_mat_compare = None
	
	## REverting train_key
	train_key = None
