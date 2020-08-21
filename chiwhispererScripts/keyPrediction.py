import pandas as pd
import numpy as np

from keras.models import load_model

import sys
sys.path.insert(0, '/xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/scr/')
import classify_3HL
import keyAccuracy

## load test data
dataPath = "/xdisk/bethard/mig2020/extra/manojgopale/AES_data/chipwhispererData/trace_key_1500/"
(_, _), (x_dev, y_dev), (_, _) = classify_3HL.getData(dataPath, 15000, 0, 1, 0) 

modelDir = "/xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/result/"
resultDir = "/xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/keyPred_files/"
logDir = "/xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/log/"

modelList = [
"chipWhisper_bs_2048_1000_700_500_50patience_3HLw_1000_700_500_100epochs_Dropout_0p2_0p2_0p2_18p10.h5",
"chipWhisper_bs_2048_200_1000_1500_50patience_3HLw_200_1000_1500_100epochs_Dropout_0p2_0p2_0p2_14p59.h5",
"chipWhisper_bs_2048_1500_400_700_50patience_3HLw_1500_400_1000_100epochs_Dropout_0p2_0p2_0p2_13p30.h5",
"chipWhisper_bs_2048_1000_1000_700_50patience_3HLw_1000_1000_700_100epochs_Dropout_0p2_0p2_0p2_12p49.h5",
"chipWhisper_bs_2048_4HL_500_1000_500_1000_50patience_4HLw_200_500_800_1000_68epochs_Dropout_0p2_0p2_0p2_0p2_12p51.h5",
"chipWhisper_bs_2048_4HL_500_1000_1500_700_50patience_4HLw_200_500_800_1000_100epochs_Dropout_0p2_0p2_0p2_0p2_13p75.h5",
"chipWhisper_bs_2048_3HL_2000_700_1500_0p2_0p2_0p2_0p2_1000epochs_50patience_3HLw_2000_1000_1500_237epochs_Dropout_0p2_0p2_0p2_15p87.h5",
"chipWhisper_bs_2048_1000epochs_100patience_6HLw_1000_700_500_500_300_300158epochs_Dropout_0p2_0p2_0p2_0p2_0p2_0p2_14p58.h5",
"chipWhisper_bs_2048_1000epochs_100patience_5HLw_1000_700_500_500_300378epochs_Dropout_0p2_0p2_0p2_0p2_0p2_17p12.h5",
"chipWhisper_bs_2048_1000epochs_100patience_7HLw_1000_700_500_500_300_300_256131epochs_Dropout_0p2_0p2_0p2_0p2_0p2_0p2__0p29p03.h5"
]

prefixList=[
"save_3HL_18p10",
"save_3HL_14p59",
"save_3HL_13p30",
"save_3HL_12p49",
"save_4HL_12p51",
"save_4HL_13p75",
"save_3HL_15p87",
"save_6HL_14p58",
"save_5HL_17p12",
"save_7HL_9p03"
]

for modelName, savePrefix in zip(modelList, prefixList):
	print("Started model= %s" %(savePrefix))
	keyAccuracy.keyAccuracy(x_dev, y_dev, modelDir, modelName, savePrefix, resultDir, logDir)
