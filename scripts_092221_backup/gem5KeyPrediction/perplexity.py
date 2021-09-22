## 1. Load model
## 2. Load dev data
## 3. Preprocess the data for 'col' or respective pre-processing
## 4. Get probas for each
## 5. Apply perplexity formula

import sys
sys.path.insert(0, '/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/')
import tensorflow as tf
import keras
from keras.models import  load_model
import newLoadData
import pandas as pd
import numpy as np
import gc

## Load model rerun49_37_dataEnsemble_combTrials_2_1_3HL_11epochs_100p00_acc_.h5
modelPath = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/../result/dataEnsemble/rerun49_37_dataEnsemble_combTrials_2_1_3HL_11epochs_100p00_acc_.h5"

model = load_model(modelPath)

dataDir = "/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/"
data=newLoadData.Data()
config = "config3p2" ##To see if we are close recall_50 ~ 95%
typeOfStd = "col"

x_dev = pd.DataFrame()
y_dev = pd.DataFrame()
x_dev, y_dev = data.getData(dataDir, config, 100, "Dev")
x_dev, y_dev= data.shuffleData(x_dev, y_dev)
print("After standardizing shapes are x_dev=%s, y_dev= %s\n" %(x_dev.shape, y_dev.shape))

meanPath = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/allConfigs/config3p1_3p2_3p3_3p4_4p1_4p2_4p3_4p4_5p1_5p2_5p3_5p4_mean.csv"
stdPath = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/allConfigs/config3p1_3p2_3p3_3p4_4p1_4p2_4p3_4p4_5p1_5p2_5p3_5p4_std.csv"
## Converting them to numpy array for standardisation
mean_pool = pd.read_csv(meanPath, header=None).to_numpy()
std_pool = pd.read_csv(stdPath, header=None).to_numpy()
print("Loaded mean and std files from\n%s\n%s\n" %(meanPath, stdPath))
## Reshaping so that it matches the standardization function and not error out
mean_pool = mean_pool.reshape(mean_pool.shape[0], )
std_pool = std_pool.reshape(std_pool.shape[0], )

##One hot the y_train and y_dev
print("Started one hot\n")
y_dev_oh = data.oneHotY(y_dev)
print("Done one hot\n")

print("After standardizing shapes are x_dev=%s, y_dev_oh= %s\n" %(x_dev.shape, y_dev_oh.shape))
## Standardize
if (typeOfStd == "col"):
	x_dev = data.stdData(x_dev.to_numpy(), mean_pool, std_pool)
	x_dev = np.where(x_dev>10, 10, x_dev)
	x_dev = np.where(x_dev<-10, -10, x_dev)
elif(typeOfStd == "row"):
	x_dev = data.stdDataRowWise(x_dev.to_numpy())

gc.collect()
print("\nGarbage collected after dev\n")

## Predict probabilities for each dev set
pred = model.predict(x_dev, batch_size=2048)

dev_actual = np.argmax(y_dev_oh, axis=1)
dev_actual = dev_actual.reshape(dev_actual.shape[0], 1) #shape should be in array form for take_along_axis

pred_prob = np.take_along_axis(pred, dev_actual, 1)
pred_prob.prod() ## Get the product of all the actual probabilities

## loop to calculate perplexity
product = 1
N = pred_prob.shape[0] #Total number of dev samples
for index in range(N):
	## Take the n'th root of each value and then multiply them
	product = product * np.power(pred_prob[index], (1/N))

perplexity = 1/product
print("perplexity of model=%s on devData= %s is %s" %(modelPath, config, perplexity))
