import sys
sys.path.insert(0, '/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/')
import classify_general
import newLoadData
import tensorflow as tf
import gc
import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Input
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np
from numpy import loadtxt
from sklearn.metrics import confusion_matrix

meanPath = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/allConfigs/config3p1_3p2_3p3_3p4_4p1_4p2_4p3_4p4_5p1_5p2_5p3_5p4_mean.csv"
stdPath = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/allConfigs/config3p1_3p2_3p3_3p4_4p1_4p2_4p3_4p4_5p1_5p2_5p3_5p4_std.csv"
## Converting them to numpy array for standardisation
mean_pool = pd.read_csv(meanPath, header=None).to_numpy()
std_pool = pd.read_csv(stdPath, header=None).to_numpy()
print("Loaded mean and std files from\n%s\n%s\n" %(meanPath, stdPath))
## Reshaping so that it matches the standardization function and not error out
mean_pool = mean_pool.reshape(mean_pool.shape[0], )
std_pool = std_pool.reshape(std_pool.shape[0], )

dataDir = "/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/"
data=newLoadData.Data()
configName = "config4p2"
devSize = 100
x_dev, y_dev = data.getData(dataDir, configName, devSize, "Dev")
x_dev, y_dev = data.shuffleData(x_dev, y_dev)
y_dev_oh = data.oneHotY(y_dev)

#Mx_dev = data.stdData(x_dev.to_numpy(), mean_pool, std_pool)
## Standardize
x_dev = data.stdData(x_dev.to_numpy(), mean_pool, std_pool)
x_dev = np.where(x_dev>10, 10, x_dev)
x_dev = np.where(x_dev<-10, -10, x_dev)

model_load = load_model("/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/../result/config4p2/rerun_config4p1_54_config4p2_run_100_3HL_14epochs_100p00_acc_.h5")
data4p2_predict = model_load.predict_classes(x_dev)
conf_4p2_4p2 = confusion_matrix(y_dev, data4p2_predict)

np.savetxt("/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/../result/config4p2/confusion_4p2_4p2.csv", conf_4p2_4p2, delimiter="," ,fmt="%.2f")

keras.backend.clear_session()

## Load 4p1
data=newLoadData.Data()
configName = "config4p1"
devSize = 100
x_dev_4p1, y_dev_4p1 = data.getData(dataDir, configName, devSize, "Dev")
x_dev_4p1, y_dev_4p1 = data.shuffleData(x_dev_4p1, y_dev_4p1)
y_dev_4p1_oh = data.oneHotY(y_dev_4p1)

x_dev_4p1 = data.stdData(x_dev_4p1.to_numpy(), mean_pool, std_pool)
x_dev_4p1 = np.where(x_dev_4p1>10, 10, x_dev_4p1)
x_dev_4p1 = np.where(x_dev_4p1<-10, -10, x_dev_4p1)

data4p1_predict = model_load.predict_classes(x_dev_4p1)
conf_4p2_4p1 = confusion_matrix(y_dev_4p1, data4p1_predict)

np.savetxt("/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/../result/config4p2/confusion_4p2_4p1.csv", conf_4p2_4p1, delimiter="," ,fmt="%.2f")
model_load_4p1 = load_model("/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/../result/config4p1/run_config4p1_54_10traces_3HL_14epochs_100p00_acc_.h5")

## Load combo model
model_load_49_43 = load_model("/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/../result/dataEnsemble/rerun49_43_dataEnsemble_3p3_5p4_4p3_NNTrials_3HL_12epochs_100p00_acc_.h5")

data4p1_predict_49_43 = model_load_49_43.predict_classes(x_dev_4p1)
conf_49_43_4p1 = confusion_matrix(y_dev_4p1, data4p1_predict_49_43)

np.savetxt("/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/../result/config4p2/confusion_4p1_49_43.csv", conf_49_43_4p1, delimiter="," ,fmt="%.2f")

