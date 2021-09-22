import tensorflow as tf
import keras
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Input
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint, EarlyStopping
import pandas as pd
import numpy as np
from numpy import loadtxt

## Load dataset
dataset = loadtxt('/Users/manojgopale/Documents/SCA/researchStuff/pima-indians-diabetes.data.csv', delimiter=',')
X = dataset[:,0:8]
y = dataset[:,8]

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=150, batch_size=10)

_, accuracy = model.evaluate(X, y)

model.save("/Users/manojgopale/Documents/SCA/researchStuff/pima-indians-diabetes.h5")


## Freeze layers and train
model2 = load_model("/Users/manojgopale/Documents/SCA/researchStuff/pima-indians-diabetes.h5")
model2.layers[3].trainable = False
for layers in model2.layers:
	print("%s: %s" %(layers.name, layers.trainable))

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model2.fit(X, y, epochs=150, batch_size=10)

_, accuracy = model.evaluate(X, y)

model2.save("/Users/manojgopale/Documents/SCA/researchStuff/pima-indians-diabetes_freeze.h5")

## Using regularizer after activation function
## https://machinelearningmastery.com/how-to-reduce-generalization-error-in-deep-neural-networks-with-activity-regularization-in-keras/
from keras.regularizers import l1

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu', activity_regularizer=l1(0.001)))
model.add(Dense(10, activation='relu', activity_regularizer=l1(0.001)))
model.add(Dense(5, activation='relu', activity_regularizer=l1(0.001)))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=150, batch_size=10)


## Plot model
from keras.utils.vis_utils import plot_model
ly1 = Dense(12, activation='relu') ##original layer from model
ly2 = Dense(12, kernel_initializer='ones', use_bias=False) ##lambda layer, initialized to ones, no bias
ly1.trainable = False
in1 = Input(shape=(8,))
x1 = ly1(in1)
x2 = ly2(x1) ##identity
mul = keras.layers.Multiply()([x1,ly2])
x3 = Dense(10, activation='relu')(mul)
x4 = Dense(5, activation='relu')(x3)
out = Dense(1, activation='sigmoid')(x4)
model = Model(inputs=in1, outputs=out)

plot_model(model, to_file='/Users/manojgopale/Documents/SCA/researchStuff/model_plot.png', show_shapes=True, show_layer_names=True)

## Using lambda
model_seq = Sequential()
model_seq.add(Dense(12, input_dim=8, activation='relu'))
model_seq.add(keras.layers.Lambda(lambda x: x*np.ones((12,1))))
model_seq.add(Dense(10, activation='relu'))
model_seq.add(Dense(5, activation='relu'))
model_seq.add(Dense(1, activation='sigmoid'))

plot_model(model, to_file='/Users/manojgopale/Documents/SCA/researchStuff/model_seq_lambda.png', show_shapes=True, show_layer_names=True)


## Add layer after output ans set the weights to identity
### Create custom contraints for weights
from keras.constraints import Constraint
import keras.backend as K

class KeepIdentity(Constraint):
	def __call__(self, W):
		w_shape = W.shape
		return K.clip(tf.math.multiply(tf.convert_to_tensor(np.identity(w_shape[0]),dtype=tf.float32), W), 0, 10)
		#return W*np.identity(w_shape[0])


## Input layer prune, to see which inputs matter the most
prune_ly0 = Dense(8, kernel_initializer='identity', kernel_constraint=KeepIdentity(), kernel_regularizer=tf.keras.regularizers.l1(0.1), use_bias=False) ##lambda layer, initialized to ones, no bias
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
model_load = load_model("/Users/manojgopale/Documents/SCA/researchStuff/pima-indians-diabetes.h5")
## model_load does not lave input layer, so the indexing is one less in model_load
model.layers[2].set_weights(model_load.layers[0].get_weights())
model.layers[4].set_weights(model_load.layers[1].get_weights())
model.layers[6].set_weights(model_load.layers[2].get_weights())
model.layers[8].set_weights(model_load.layers[3].get_weights())

plot_model(model, to_file='/Users/manojgopale/Documents/SCA/researchStuff/model_identity.png', show_shapes=True, show_layer_names=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=150, batch_size=10)


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
## Reset graphs in network
keras.backend.clear_session()

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
