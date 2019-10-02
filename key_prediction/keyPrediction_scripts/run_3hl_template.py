import sys
sys.path.insert(0, '/extra/manojgopale/AES_data/keyPrediction_scripts/')
import classify_3HL 
import time

from optparse import OptionParser

parser = OptionParser()
parser.add_option('-c','--config',
									action = 'store', type='string', dest='config', default = 'config3p1')
parser.add_option('--trainSize',
									action = 'store', type='int', dest='trainSize', default = 15000)
parser.add_option('--resultDir',
									action = 'store', type='string', dest='resultDir', default = '/extra/manojgopale/AES_data/config3p1_15ktraining/result_new/')
parser.add_option('--batchSize',
									action = 'store', type='int', dest='batchSize', default = 2048)
parser.add_option('--modelName',
									action = 'store', type='string', dest='modelName', default = 'defaultName')

(options, args) = parser.parse_args()

########
config = options.config
trainSize = options.trainSize
resultDir = options.resultDir
batchSize = options.batchSize
modelName = options.modelName

dataDir = "/xdisk/manojgopale/data_csv/" + config + "/"
trainData, devData, testData = classify.getData(dataDir, trainSize)

x_train, y_train_oh = trainData
x_dev, y_dev_oh = devData
x_test, y_test_oh = testData

## Instantiate the model and test, dev and training sets
##resultDir = "/extra/manojgopale/AES_data/config3p1_15ktraining/result_new"
##modelName = "m_newscript"

t0_time = time.time()
classifier = classify.Classifier(resultDir, modelName, x_train, y_train_oh, x_dev, y_dev_oh, x_test, y_test_oh)
t1_time = time.time()
print("\nTime to load the dataset in python for training is %s seconds\n" %(t1_time-t0_time))

## Train the model
startTime = time.time()
classifier.train(batchSize)
endTime = time.time()
trainTime = endTime - startTime
print("\nTime to train %s with batchSize= %s is %s seconds\n" %(config, batchSize, trainTime))

## Evaluate
classifier.evaluate()

##Save the model
classifier.saveModel()
