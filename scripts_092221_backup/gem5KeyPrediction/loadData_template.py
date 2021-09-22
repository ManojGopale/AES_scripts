import sys
sys.path.insert(0, '/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/')
from newLoadData import  Data

data = Data()
dataPath="/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/"
configName = "config3p2"
tracesPerKey = 28000
dataSetType = "Train"
x_train, y_train = data.getData(dataPath, configName, tracesPerKey, dataSetType)
##~86Gb for x_train, 1224Mb for y_train
sys.getsizeof(x_train)
sys.getsizeof(y_train)

#Mx_train, y_train = data.shuffleData(x_train, y_train)
#My_train_oh = data.oneHotY(y_train)
#Mx_train_mean, x_train_std = data.getStdParam(x_train)
