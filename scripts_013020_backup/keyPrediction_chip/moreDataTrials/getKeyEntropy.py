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

from sklearn.utils import shuffle
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import clone_model
from keras.models import load_model
from collections import Counter

def  getKeyEntropoy(modelList)
	for index, modelPath in enumerate(modelList):
		model = load_model(modelPath)
		model_score = model.evaluate(x_dev, y_dev_oh, batch_size=2048)
		y_pred = np.argmax(model.predict(x_dev, batch_size=2048), axis=1)

		output_predict = np.vstack((np.argmax(y_dev_oh, axis=1), y_pred)).T

		df = pd.DataFrame(data=output_predict)

		error_df = df[df[0]!=df[1]].astype('category')
		error_df[2] = error_df[0].astype(str).str.cat(error_df[1].astype(str), sep="-")
		
		totalCount = df[0].count()
		errorCount = error_df[2].count()
		accuracy = ((df[0].count()-error_df[2].count())/df[0].count())*100

		##Checking for entopy calculations of the predictions
		pred = model.predict(x_dev, batch_size=2048)
		
		## np.argsort(-pred) gets the order in which the indexes will be arranged 
		## np.argsort() of above will return the rank of the indexes with their values in the coressponding indexes
		rank = np.argsort(np.argsort(-pred))
		
		## Get the actual predictions from y_dev, need to convert one-hot to actual numbers
		dev_actual = np.argmax(y_dev_oh, axis=1)
		
		## Get the prediction ranks for each prediction
		prediction_ranks = rank[np.arange(len(dev_actual)), dev_actual]

		## Get thte top 5 ransk of each prediction
		## this creates a df with 5 rows per key, we will have to use groupby and agg to get list of top 5 predictions
		rank_df = pd.DataFrame(data=np.argwhere(rank<5))
		getRank_df = rank_df.groupby([0]).agg(lambda x: list(x))
		getRank_df[2] = dev_actual

		##REname columns
		getRank_df.rename(columns={1: "topRank", 2: "actualRank"}, inplace=True)

		## Create a flat list for each key predictions
		getRank_df.groupby("actualRank").agg(lambda x: [z for y in x for z in y])


		## Get the top most frequently predicted keys and its count
		## We can get the other top list by changing most_common number
		aggRank = getRank_df.groupby("actualRank", as_index=False).agg(lambda x: Counter([z for y in x for z in y]).most_common(1))

		## get the accuracy of each key
		## this will put 0 where the aggAccu is not for the actual key
		aggRank["aggAcc"] = aggRank.apply(lambda x: 0 if x.topRank[0][0] != x.actualRank else x.topRank[0][1]/10, axis=1)

		## print where the agg is not the actual key
		print ("keys where the agg is not the acutal key\n%s" %(aggRank["aggAcc"][aggRank["aggAcc"] == 0]))


		## Save aggRank to a csv and see results afterwards


		## Append all the ranks to the df

		keyAcc = pd.DataFrame(columns={'key', 'acc'})
		
		for key in range(255):
			totalKey = df[0][df[0]==key].count()
			keyErrors = error_df[0][error_df[0]==key].count()
			acc = ((totalKey-keyErrors)/totalKey)*100
			keyAcc.loc[key] = {'key': key, 'acc': acc}



