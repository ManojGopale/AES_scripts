import pandas as pd
import numpy as np

from keras.models import load_model

import sys
sys.path.insert(0, '/xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/scr/')
import classify_3HL_28000TrainSize
import keyAccuracy

## load test data
dataPath = "/xdisk/rlysecky/manojgopale/extra/chipWhisperer_data/trace_key_1500/"
moreData = "/xdisk/rlysecky/manojgopale/extra/chipWhisperer_data/trace_key_1500_1/"
(_, _), (x_dev, y_dev), (_, _) = classify_3HL_28000TrainSize.getData(dataPath, moreData, 28000, 1, 1, 1)

modelDir = "/xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/result/"
resultDir = "/xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/keyPredFiles/"
logDir = "/xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/log/"

modelList = [
"chipWhisper_3HL_28000TS_20480BS_newNorm_train_metrics_3HLw_1000_700_500_73epochs_Dropout_0p2_0p2_0p2_41p42.h5"
]

prefixList=[
"3hl_41p42_newNorm_train_metric",
]

for modelName, savePrefix in zip(modelList, prefixList):
	print("Started model= %s" %(savePrefix))
	keyAccuracy.keyAccuracy(x_dev, y_dev, modelDir, modelName, savePrefix, resultDir, logDir)
