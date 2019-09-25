import sys
sys.path.insert(0, '/extra/manojgopale/AES_data/')
import classify_1HL 

def runConfig(dataDir, trainDataSize, resultDir, modelName, hiddenSize, batchSize):
	trainData, devData, testData = classify_1HL.getData(dataDir, trainDataSize)
	
	x_train, y_train_oh = trainData
	x_dev, y_dev_oh = devData
	x_test, y_test_oh = testData
	
	## Instantiate the model and test, dev and training sets
	classifier = classify_1HL.Classifier(resultDir, modelName, x_train, y_train_oh, x_dev, y_dev_oh, x_test, y_test_oh, hiddenSize)
	
	## Train the model
	classifier.train(batchSize)
	
	## Evaluate
	classifier.evaluate()
	
	##Save the model
	classifier.saveModel()


dataDir = "/xdisk/manojgopale/data_csv/config5p4/"
resultDir = "/extra/manojgopale/AES_data/config5p4_15ktraining/result_new"
trainDataSize = 15000

#M## 100 hidden neurons
#MmodelName = "model_1hl_100neurons_1"
#MhiddenSize = 100
#MbatchSize = 2048
#M
#MrunConfig(dataDir, trainDataSize, resultDir, modelName, hiddenSize, batchSize)
#M
#M## 500 hidden neurons
#MmodelName = "model_1hl_500neurons_2"
#MhiddenSize = 500
#MbatchSize = 2048
#M
#MrunConfig(dataDir, trainDataSize, resultDir, modelName, hiddenSize, batchSize)

## 1000 hidden neurons
modelName = "model_1hl_1000neurons_3"
hiddenSize = 1000
batchSize = 2048

runConfig(dataDir, trainDataSize, resultDir, modelName, hiddenSize, batchSize)

## 21500 hidden neurons
modelName = "model_1hl_1500neurons_4"
hiddenSize = 1500
batchSize = 2048

runConfig(dataDir, trainDataSize, resultDir, modelName, hiddenSize, batchSize)

## 2000 hidden neurons
modelName = "model_1hl_2000neurons_5"
hiddenSize = 2000
batchSize = 2048

runConfig(dataDir, trainDataSize, resultDir, modelName, hiddenSize, batchSize)
