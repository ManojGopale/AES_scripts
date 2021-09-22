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

from keras.constraints import Constraint
import keras.backend as K

class KeepIdentity(Constraint):
	def __call__(self, W):
		w_shape = W.shape
		return K.clip(tf.math.multiply(tf.convert_to_tensor(np.identity(w_shape[0]),dtype=tf.float32), W), 0, 10)
		#return W*np.identity(w_shape[0])

## Load dataset on which the pruning is to be performed
## Could be a dataset differnet than what it was trained on
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

for index in range(5):
	plt.plot(plt_x, x_dev.iloc[index], 'r')
	figName = resultDir + "/" + configName + "/images_debug/" + modelName + "_dev_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_preStd_key" + str(y_dev.iloc[index].values[0]) + ".png"
	plt.savefig(figName)
	plt.close()

#Mx_dev = data.stdData(x_dev.to_numpy(), mean_pool, std_pool)
## Standardize
if (typeOfStd == "col"):
	x_dev = data.stdData(x_dev.to_numpy(), mean_pool, std_pool)
	x_dev = np.where(x_dev>10, 10, x_dev)
	x_dev = np.where(x_dev<-10, -10, x_dev)
elif(typeOfStd == "row"):
	x_dev = data.stdDataRowWise(x_dev.to_numpy())

## Commenting out the saving to file, because the error was not in data and it is huge is size
##Mdev_std_path = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/" + configName + "/dev_std_3p1_3p2.csv"
##Mnp.savetxt(dev_std_path, x_dev, delimiter=",")
for index in range(5):
	plt.plot(plt_x, x_dev[index], 'g')
	figName = resultDir + "/" + configName + "/images_debug/" + modelName + "_dev_" + str(time.strftime("%Y%m%d-%H%M%S")) + "_postStd_key" + str(y_dev.iloc[index].values[0]) + ".png"
	plt.savefig(figName)
	plt.close()

gc.collect()
print("\nGarbage collected after dev\n")

#Mdataset = loadtxt('/Users/manojgopale/Documents/SCA/researchStuff/pima-indians-diabetes.data.csv', delimiter=',')
#MX = dataset[:,0:8]
#My = dataset[:,8]

## Reset graphs in network
keras.backend.clear_session()

## Input layer prune, to see which inputs matter the most
prune_ly0 = Dense(1500, kernel_initializer='identity', kernel_constraint=KeepIdentity(), kernel_regularizer=tf.keras.regularizers.l1(0.1), use_bias=False) ##lambda layer, initialized to ones, no bias
prune_ly1 = Dense(32, kernel_initializer='identity', kernel_constraint=KeepIdentity(), kernel_regularizer=tf.keras.regularizers.l1(0.1), use_bias=False) ##lambda layer, initialized to ones, no bias
prune_ly2 = Dense(64, kernel_initializer='identity', kernel_constraint=KeepIdentity(), kernel_regularizer=tf.keras.regularizers.l1(0.01), use_bias=False) ##lambda layer, initialized to ones, no bias
prune_ly3 = Dense(256, kernel_initializer='identity', kernel_constraint=KeepIdentity(), kernel_regularizer=tf.keras.regularizers.l1(0.01), use_bias=False) ##lambda layer, initialized to ones, no bias
#ly2 = Dense(12, kernel_initializer='identity',  use_bias=False) ##lambda layer, initialized to ones, no bias

## Create model
in1 = Input(shape=(1500,))
prune0 = prune_ly0(in1)
x1 = Dense(32, activation='elu')(prune0)
prune1 = prune_ly1(x1)
x2 = Dense(64, activation='relu')(prune1)
prune2 = prune_ly2(x2)
x3 = Dense(256, activation='tanh')(prune2)
prune3 = prune_ly3(x3)
out = Dense(256, activation='softmax')(prune3)
model = Model(inputs=in1, outputs=out)

## Freeze layers.
model.layers[2].trainable = False
model.layers[4].trainable = False
model.layers[6].trainable = False
model.layers[8].trainable = False

## for input only
#Mmodel.layers[1].trainable = False ##Input layer is only trainable
model.layers[3].trainable = False
model.layers[5].trainable = False
model.layers[7].trainable = False

## Set weights from previously trained model
## Load the target model that we want to test.
model_load = load_model("/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/../result/config4p2/rerun_config4p1_54_config4p2_run_100_3HL_14epochs_100p00_acc_.h5")
_,accuracy_load = model_load.evaluate(x_dev, y_dev_oh)
print("accuracy of loaded model on data is %s\n" %(accuracy_load))

## model_load does not lave input layer, so the indexing is one less in model_load
## model_load has alternate dropout layers, so the indexing in multiple of '2' in set_weight
model.layers[2].set_weights(model_load.layers[0].get_weights())
model.layers[4].set_weights(model_load.layers[2].get_weights())
model.layers[6].set_weights(model_load.layers[4].get_weights())
model.layers[8].set_weights(model_load.layers[6].get_weights())

model.compile(loss='categorical_crossentropy', metrics=['categorical_accuracy'], optimizer="adam")
_,accuracy_prior = model.evaluate(x_dev, y_dev_oh)
print("accuracy prior to pruning of model on data is %s\n" %(accuracy_prior))

logFile = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/../result/config4p2/model_4p2_rerun54_100_on_4p2.log"
csv_logger = CSVLogger(logFile, append=True, separator="\t")
history = model.fit(x_dev, y_dev_oh, batch_size= 64, epochs=10, verbose=1, shuffle= True, validation_data=(x_dev, y_dev_oh), callbacks=[csv_logger])

## Save the model for future reference

## Save the pruned diagonal for future comparisons
prune_diag = np.diagonal(model.layers[1].get_weights()[0])
prune_diag_gt_zero = np.argwhere(prune_diag>0)
## Once save we can copy the csv in excel and compare to start
np.savetxt("/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/../result/config4p2/model_4p2_rerun54_100_on_4p2.csv", prune_diag, delimiter=",", fmt="%2.4f")

## Load 4p1 data, with 4p2 model
## accuracy is 0.4%
## Now change the prune layer to match the one in 4p1_4p1 prune
from numpy import genfromtxt

prune_diag_4p1_on_4p1 = genfromtxt('/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/../result/config4p1/model_4p1_rerun54_100_on_4p1.csv', delimiter=',')
prune_wght_4p1_on_4p1 = model.layers[1].get_weights()[0]*prune_diag_4p1_on_4p1
model.layers[1].set_weights([prune_wght_4p1_on_4p1])


## Can we freeze the input prune layers from one model to another model and see any improvements on the inter-arch portability? 
## The case to see is that each prune layer is specific to the model built, but we do have architecture similarity in the models, so a worth while experimentation

## train the base model with normalized inputs
model_seq = Sequential()
model_seq.add(Dense(12, input_dim=8, activation='relu'))
model_seq.add(Dense(10, activation='relu'))
model_seq.add(Dense(5, activation='relu'))
model_seq.add(Dense(1, activation='sigmoid'))

## If we want to test the model's dependency on inputs, I think we should normalize the inputs first.
X_norm = ((X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0)))
model_seq.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_seq.fit(X_norm, y, epochs=150, batch_size=10)

_, accuracy = model_seq.evaluate(X_norm, y)

model_seq.save("/Users/manojgopale/Documents/SCA/researchStuff/pima-indians-diabetes_normInputs.h5")

## Now create prune model and test which inputs matter most

## Input layer prune, to see which inputs matter the most
prune_ly0 = Dense(8, kernel_initializer='identity', kernel_constraint=KeepIdentity(), kernel_regularizer=tf.keras.regularizers.l1(0.09), use_bias=False) ##lambda layer, initialized to ones, no bias
prune_ly1 = Dense(12, kernel_initializer='identity', kernel_constraint=KeepIdentity(), kernel_regularizer=tf.keras.regularizers.l1(0.01), use_bias=False) ##lambda layer, initialized to ones, no bias
prune_ly2 = Dense(10, kernel_initializer='identity', kernel_constraint=KeepIdentity(), kernel_regularizer=tf.keras.regularizers.l1(0.01), use_bias=False) ##lambda layer, initialized to ones, no bias
prune_ly3 = Dense(5, kernel_initializer='identity', kernel_constraint=KeepIdentity(), kernel_regularizer=tf.keras.regularizers.l1(0.01), use_bias=False) ##lambda layer, initialized to ones, no bias
#ly2 = Dense(12, kernel_initializer='identity',  use_bias=False) ##lambda layer, initialized to ones, no bias

## Create model
in1 = Input(shape=(8,))
prune0 = prune_ly0(in1)
x1 = Dense(12, activation='relu')(prune0)
prune1 = prune_ly1(x1)
x3 = Dense(10, activation='relu')(prune1)
prune2 = prune_ly2(x3)
x4 = Dense(5, activation='relu')(prune2)
prune3 = prune_ly3(x4)
out = Dense(1, activation='sigmoid')(prune3)
model = Model(inputs=in1, outputs=out)

## Freeze layers.
model.layers[2].trainable = False
model.layers[4].trainable = False
model.layers[6].trainable = False
model.layers[8].trainable = False

## Set weights from previously trained model
model_load = load_model("/Users/manojgopale/Documents/SCA/researchStuff/pima-indians-diabetes_normInputs.h5")
## model_load does not lave input layer, so the indexing is one less in model_load
model.layers[2].set_weights(model_load.layers[0].get_weights())
model.layers[4].set_weights(model_load.layers[1].get_weights())
model.layers[6].set_weights(model_load.layers[2].get_weights())
model.layers[8].set_weights(model_load.layers[3].get_weights())

plot_model(model, to_file='/Users/manojgopale/Documents/SCA/researchStuff/model_identity.png', show_shapes=True, show_layer_names=True)

## Earlystopping on first look seems to not be a good fit for this, becasue the accuracy is going to decrease with l1 regularization
## The problem is with more epochs all the prune layers might go to '0'.
## WE can have early stopping after few epochs.
## Or we can train for 10 epochs and then apply earlyStop
## The model has already trained and weights fixed. So maybe we need to keep gamma low so that the weights dont end up going to '0'
earlyStop = EarlyStopping(monitor='acc', patience=5, mode='auto', verbose=1, restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

X_norm = ((X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0)))
model.fit(X_norm, y, epochs=100, batch_size=10)
#model.fit(X_norm, y, epochs=100, batch_size=10,callbacks=[earlyStop])

## To see if the model is actually similar use confusion matrix

from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt

## ConfusionMatrix(y_true,y_pred)
predict = model.predict(X_norm)
predict = np.where(predict<0.5,0,1)
cm = ConfusionMatrix(y, predict[:,0])
#Predicted  False  True  __all__
#Actual                         
#False        458    42      500
#True         155   113      268
#__all__      613   155      768


pred_old = np.where(model_load.predict(X_norm)<0.5, 0,1)
cm_old = ConfusionMatrix(y, pred_old[:,0])
#Predicted  False  True  __all__
#Actual
#False        463    37      500
#True         118   150      268
#__all__      581   187      768


## Fix all other prune layers except the 1st one, to see which inputs affect the output most
model.layers[3].trainable = False
model.layers[5].trainable = False
model.layers[7].trainable = False

#>>> np.diagonal(model.layers[1].get_weights()[0])
#array([0.        , 1.1891226 , 0.        , 0.        , 0.        ,
#       0.76996773, 0.        , 0.13022116], dtype=float32)

#Predicted  False  True  __all__
#Actual                         
#False        445    55      500
#True         121   147      268
#__all__      566   202      768


## Get outputs at each layer
## https://stackoverflow.com/questions/52039079/how-to-get-intermediate-output-when-using-tf-keras-application

model_4p1 = load_model("/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/../result/config4p1/rerun_config4p1_54_config4p1_run_100_3HL_16epochs_100p00_acc_.h5")
model_4p2 = load_model("/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/../result/config4p2/rerun_config4p1_54_config4p2_run_100_3HL_14epochs_100p00_acc_.h5")

get_first_layer_4p1 = K.function([model_4p1.layers[0].input, K.learning_phase()], [model_4p1.layers[1].output])
get_first_layer_4p2 = K.function([model_4p2.layers[0].input, K.learning_phase()], [model_4p2.layers[1].output])

get_last_layer_4p1 = K.function([model_4p1.layers[0].input, K.learning_phase()], [model_4p1.layers[6].output])
get_last_layer_4p2 = K.function([model_4p2.layers[0].input, K.learning_phase()], [model_4p2.layers[6].output])


## This generates the output at 1st hidden layer for all the inputs 256000x32
layer1 = get_first_layer_4p1([x_dev, 0])
layer1_24 = get_first_layer_4p1([x_dev.loc[24,:].to_numpy().reshape(1,1500), 0])

layer2 = get_first_layer_4p2([x_dev_4p2, 0])
layer2_24 = get_first_layer_4p2([x_dev_4p2.loc[43,:].to_numpy().reshape(1,1500), 0])


