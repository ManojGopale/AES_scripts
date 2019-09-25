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

runDir = "/extra/manojgopale/AES_data/config5p4_15ktraining/"
dataDir = runDir + "data/"
resultDir = runDir + "result/"

train0Str = dataDir + "aesData_config9_Train_0.csv"
dev0Str = dataDir + "aesData_config9_Dev_0.csv"
test0Str = dataDir + "aesData_config9_Test_0.csv"

x_train_0_63, y_train_0_63 = process_inputs(train0Str)
x_dev_0_63, y_dev_0_63 = process_inputs(dev0Str)
x_test_0_63, y_test_0_63 = process_inputs(test0Str)

print("Loaded 0-63 train, dev and test and processed them")

train1Str = dataDir + "aesData_config9_Train_1.csv"
dev1Str = dataDir + "aesData_config9_Dev_1.csv"
test1Str = dataDir + "aesData_config9_Test_1.csv"

x_train_64_127, y_train_64_127 = process_inputs(train1Str)
x_dev_64_127, y_dev_64_127 = process_inputs(dev1Str)
x_test_64_127, y_test_64_127 = process_inputs(test1Str)

print("Loaded 64-127 train, dev and test and processed them")

train2Str = dataDir + "aesData_config9_Train_2.csv"
dev2Str = dataDir + "aesData_config9_Dev_2.csv"
test2Str = dataDir + "aesData_config9_Test_2.csv"

x_train_128_191, y_train_128_191 = process_inputs(train2Str)
x_dev_128_191, y_dev_128_191 = process_inputs(dev2Str)
x_test_128_191, y_test_128_191 = process_inputs(test2Str)

print("Loaded 128-191 train, dev and test and processed them")

train3Str = dataDir + "aesData_config9_Train_3.csv"
dev3Str = dataDir + "aesData_config9_Dev_3.csv"
test3Str = dataDir + "aesData_config9_Test_3.csv"

x_train_192_255, y_train_192_255 = process_inputs(train3Str)
x_dev_192_255, y_dev_192_255 = process_inputs(dev3Str)
x_test_192_255, y_test_192_255 = process_inputs(test3Str)

print("Loaded 192-255 train, dev and test and processed them")

## Concatinating all the datasets before onehot assignment

#x_train_0_255 = np.concatenate((x_train_0_63, x_train_64_127, x_train_128_191, x_train_192_255), 0)
#y_train_0_255 = np.concatenate((y_train_0_63, y_train_64_127, y_train_128_191, y_train_192_255), 0)

x_train_0_255 = np.concatenate((x_train_64_127, x_train_0_63, x_train_192_255, x_train_128_191), 0)
y_train_0_255 = np.concatenate((y_train_64_127, y_train_0_63, y_train_192_255, y_train_128_191), 0)

## Clearing variables that are not required for freeing up memory
x_train_0_63 = None
x_train_64_127 = None
x_train_128_191 = None
x_train_192_255 = None

y_train_0_63 = None
y_train_64_127 = None
y_train_128_191 = None
y_train_191_255 = None

#########

x_dev_0_255 = np.concatenate((x_dev_0_63, x_dev_64_127, x_dev_128_191, x_dev_192_255), 0)
y_dev_0_255 = np.concatenate((y_dev_0_63, y_dev_64_127, y_dev_128_191, y_dev_192_255), 0)

## Clearing variables that are not required for freeing up memory
x_dev_0_63 = None
x_dev_64_127 = None
x_dev_128_191 = None
x_dev_192_255 = None

y_dev_0_63 = None
y_dev_64_127 = None
y_dev_128_191 = None
y_dev_191_255 = None

#########

x_test_0_255 = np.concatenate((x_test_0_63, x_test_64_127, x_test_128_191, x_test_192_255), 0)
y_test_0_255 = np.concatenate((y_test_0_63, y_test_64_127, y_test_128_191, y_test_192_255), 0)

## Clearing variables that are not required for freeing up memory
x_test_0_63 = None
x_test_64_127 = None
x_test_128_191 = None
x_test_192_255 = None

y_test_0_63 = None
y_test_64_127 = None
y_test_128_191 = None
y_test_191_255 = None

print("Concatenation of all variables done")
########

## Alternate way 
#clearList = ['x_train_0_63', 'x_train_64_127', 'x_train_128_191', 'x_train_192_255', 'y_train_0_63', 'y_train_64_127', 'y_train_128_191', 'y_train_191_255']
#
#for var in clearList:
#	exec("%s=None" %(str(var)))

print("CLeared non required variables")

## One hot assignment
n_classes = 256
y_train_0_255_oh = np_utils.to_categorical(y_train_0_255, n_classes)
y_dev_0_255_oh = np_utils.to_categorical(y_dev_0_255, n_classes)
y_test_0_255_oh = np_utils.to_categorical(y_test_0_255, n_classes)

print("One-hot encoded for outputs")

#Model1
#logFile = resultDir + 'm1.log'
#csv_logger = CSVLogger(logFile, append=True, separator="\t")

#model1 = Sequential()
#
#model1.add(Dense(100, input_shape=(1361,)))
#model1.add(Activation('relu'))                            
#
#model1.add(Dense(100))
#model1.add(Activation('relu'))
#
#model1.add(Dense(256))
#model1.add(Activation('softmax'))
#
#model1.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')
#history_1 = model1.fit(x_train_0_255, y_train_0_255_oh, batch_size= 160000, epochs=30, verbose=1, shuffle= True, validation_data=(x_dev_0_255, y_dev_0_255_oh), callbacks=[csv_logger])
#
#model1_score = model1.evaluate(x_test_0_255, y_test_0_255_oh, batch_size=160000)
#
#print("model1_score= %f" %(model1_score[1]))
#
#saveStr = resultDir + 'm1_1HL_30epoch_' + f'{model1_score[1]*100:.2f}'.replace('.', 'p') + '.h5'
#model1.save(saveStr)


## Model2
#logDir = runDir + "log/{}".format(datetime.now().strftime("%m-%d-%y__%H_%M")) 
#tensorboard = TensorBoard(log_dir=logDir, histogram_freq=0, batch_size=256, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
#
#logFile = resultDir + 'm2_newTrial.log'
#csv_logger = CSVLogger(logFile, append=True, separator="\t")
#
#earlyStop = EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=1)
#
#Epochs = 40
#dropOut = 0.2
#
#model2 = Sequential()
#
#model2.add(Dense(100, input_shape=(1361,)))
#model2.add(Activation('relu'))                            
#
#model2.add(Dropout(dropOut))
#model2.add(Dense(256))
#model2.add(Activation('softmax'))
#
#model2.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')
#history_2 = model2.fit(x_train_0_255, y_train_0_255_oh, batch_size= 160000, epochs=Epochs, verbose=1, shuffle= True, validation_data=(x_dev_0_255, y_dev_0_255_oh), callbacks=[csv_logger, tensorboard, earlyStop])
#
#model2_score = model2.evaluate(x_test_0_255, y_test_0_255_oh, batch_size=160000)
#
#print("model2_score= %f" %(model2_score[1]))
#
#saveStr = resultDir + 'm2_newTrial_1HL_' + str(Epochs) + 'epochs_' + str(dropOut).replace('.', 'p') + 'Dropout_' + f'{model2_score[1]*100:.2f}'.replace('.', 'p') + '.h5'
#model2.save(saveStr)
#
### Model 8
### Tensorboard
#from keras.callbacks import TensorBoard
#from datetime import date, datetime
#
##logDir = runDir + "log/{}".format(datetime.now().strftime("%m-%d-%y__%H_%M")) 
##tensorboard = TensorBoard(log_dir=logDir, histogram_freq=0, batch_size=256, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
#
#logFile = resultDir + 'm8_newTrial_run2.log'
#csv_logger = CSVLogger(logFile, append=True, separator="\t")
#
##earlyStop = EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=1)
#
#Epochs = 1000
#dropOut = 0.2
#
#model8 = Sequential()
#
#model8.add(Dense(500, input_shape=(1361,)))
#model8.add(Activation('relu'))                            
#
#model8.add(Dropout(dropOut))
#model8.add(Dense(500))
#
#model8.add(Dropout(dropOut))
#model8.add(Dense(256))
#
#model8.add(Dropout(dropOut))
#model8.add(Dense(256))
#model8.add(Activation('softmax'))
#
#model8.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')
#history_8 = model8.fit(x_train_0_255, y_train_0_255_oh, batch_size= 160000, epochs=Epochs, verbose=1, shuffle= True, validation_data=(x_dev_0_255, y_dev_0_255_oh), callbacks=[csv_logger])
#
##history_8 = model8.fit(x_train_0_255, y_train_0_255_oh, batch_size= 160000, epochs=Epochs, verbose=1, shuffle= True, validation_data=(x_dev_0_255, y_dev_0_255_oh), callbacks=[csv_logger, earlyStop, tensorboard])
#
#model8_score = model8.evaluate(x_test_0_255, y_test_0_255_oh, batch_size=160000)
#
#print("model8_score= %f" %(model8_score[1]))
#
#saveStr = resultDir + 'm8_3HLw_500_500_256_noDrop_run2_' + str(Epochs) + 'epochs_' + str(dropOut).replace('.', 'p') + 'Dropout_' + '{0:.2f}'.format(model8_score[1]*100).replace('.', 'p') + '.h5'
#model8.save(saveStr)

### With new EArly stopping
#import sys
#sys.path.insert(0, '/extra/manojgopale/AES_data/')
#
#####################
######## Note #######
#####################
## EarlyStop  is now in newCallbacks folder, check config3p1_15traininig/scr for reference
#
#import EarlyStopNew
#
#from keras.callbacks import TensorBoard
#from datetime import date, datetime
#
##logDir = runDir + "log/{}".format(datetime.now().strftime("%m-%d-%y__%H_%M")) 
##tensorboard = TensorBoard(log_dir=logDir, histogram_freq=1, batch_size=256, write_graph=True, write_grads=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
#
#logFile = resultDir + 'm9_newTrial_withHist.log'
#csv_logger = CSVLogger(logFile, append=True, separator="\t")
#
#earlyStop = EarlyStopNew.EarlyStopNew(monitor='val_categorical_accuracy', patience=5, mode='auto', verbose=1)
#
#filePath = resultDir + 'm9_checkPoint_best_model.hdf5'
#checkPoint = ModelCheckpoint(filePath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True)
#
#Epochs = 1000
#dropOut = 0.2
#
#model9 = Sequential()
#
#model9.add(Dense(500, input_shape=(1361,)))
#model9.add(Activation('relu'))                            
#
#model9.add(Dropout(dropOut))
#model9.add(Dense(500))
#
#model9.add(Dropout(dropOut))
#model9.add(Dense(256))
#
#model9.add(Dropout(dropOut))
#model9.add(Dense(256))
#model9.add(Activation('softmax'))
#
#model9.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')
#history_9 = model9.fit(x_train_0_255, y_train_0_255_oh, batch_size= 160000, epochs=Epochs, verbose=1, shuffle= True, validation_data=(x_dev_0_255, y_dev_0_255_oh), callbacks=[csv_logger, earlyStop, checkPoint])
#
#model9_score = model9.evaluate(x_test_0_255, y_test_0_255_oh, batch_size=160000)
#
#print("model9_score= %f" %(model9_score[1]))
#
#saveStr = resultDir + 'm9_3HLw_500_500_256_' + str(Epochs) + 'epochs_' + str(dropOut).replace('.', 'p') + 'Dropout_' + '{0:.2f}'.format(model9_score[1]*100).replace('.', 'p') + '.h5'
#model9.save(saveStr)

### Model 10
## Tensorboard
from keras.callbacks import TensorBoard
from datetime import date, datetime

#logDir = runDir + "log/{}".format(datetime.now().strftime("%m-%d-%y__%H_%M")) 
#tensorboard = TensorBoard(log_dir=logDir, histogram_freq=0, batch_size=256, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

logFile = resultDir + 'm10_newTrial.log'
csv_logger = CSVLogger(logFile, append=True, separator="\t")

earlyStop = newCallBacks.EarlyStopNew(monitor='val_categorical_accuracy', patience=5, mode='auto', verbose=1)

filePath = resultDir + 'm10_checkPoint_best_model.hdf5'
## This file will include the epoch number when it gets saved.
#repeatingFile = resultDir + 'm10_{epoch:02d}_epoch_5p2_acc_{acc_5p2:.2f}.hdf5'
repeatingFile = resultDir + 'm10_{epoch:02d}.hdf5'

## By default the every_10epochs will save the model at every 10 epochs
## This is for evaluating 5.2 accuracy while training for 3.1. You can skip mentioning the 2 test files and it should work
##checkPoint = newCallBacks.ModelCheckpoint_every_10epochs(filePath, repeatingFile, x_5p2_test_0_255, y_5p2_test_0_255_oh , monitor='val_categorical_accuracy', verbose=1, save_best_only=True, every_10epochs=True)

checkPoint = newCallBacks.ModelCheckpoint_every_10epochs(filePath, repeatingFile, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, every_10epochs=True)

Epochs = 1000
dropOut = 0.2

model10 = Sequential()

model10.add(Dense(500, input_shape=(1361,)))
model10.add(Activation('relu'))                            

model10.add(Dropout(dropOut))
model10.add(Dense(500))

model10.add(Dropout(dropOut))
model10.add(Dense(256))

model10.add(Dropout(dropOut))
model10.add(Dense(256))
model10.add(Activation('softmax'))

model10.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer='adam')
history_10 = model10.fit(x_train_0_255, y_train_0_255_oh, batch_size= 2048, epochs=Epochs, verbose=1, shuffle= True, validation_data=(x_dev_0_255, y_dev_0_255_oh), callbacks=[csv_logger, checkPoint, earlyStop])

model10_score = model10.evaluate(x_test_0_255, y_test_0_255_oh, batch_size=2048)

print("model10_score= %f" %(model10_score[1]))

## Saving with last epoch count
saveStr = resultDir + 'm10_3HLw_500_500_256_noDrop_' + str(history_10.epoch[-1]+1) + 'epochs_' + str(dropOut).replace('.', 'p') + 'Dropout_' + '{0:.2f}'.format(model10_score[1]*100).replace('.', 'p') + '.h5'
model10.save(saveStr)
