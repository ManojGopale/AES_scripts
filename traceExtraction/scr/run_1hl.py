## This file will process the matlab data to get back train, dev and test data
import sys
sys.path.insert(0, '/extra/manojgopale/AES_data/traceExtraction/scr/classify_1HL.py')
import classify_1HL

from optparse import OptionParser

parser = OptionParser()
parser.add_option('-c','--config',
									action = 'store', type='string', dest='config', default = 'config3p1')
parser.add_option('-b', '--batchCount',
									action = 'store', type='int', dest='batchCount', default = 128)
parser.add_option('-h', '--hiddenSize',
									action = 'store', type='int', dest='hiddenSize', default = 1000)
parser.add_option('--modelName',
									action = 'store', type='string', dest='modelName', default = 'defaultName')

(options, args) = parser.parse_args()


## Collect data
trainDataSize = 4000 ## There are 4000 traces per key
config = options.config
modelName = options.modelName
hiddenSize = options.hiddenSize
resultDir = "/xdisk/manojgopale/AES/dataCollection/processedData/run_1_per_key/result/" + config + "/"

trainData, devData, testData = classify_1HL.getData(config, trainDataSize)

x_train, y_train = trainData
x_dev, y_dev = devData
x_test, y_test = testData

## Instantiate the model and test, dev and training sets
classifier = classify_1HL.Classifier(resultDir, modelName, x_train, y_train, x_dev, y_dev, x_test, y_test, hiddenSize)

## Train the model
classifier.train(batchSize)

## Evaluate
classifier.evaluate()

##Save the model
classifier.saveModel()


