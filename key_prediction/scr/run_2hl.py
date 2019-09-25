import sys
sys.path.insert(0, '/extra/manojgopale/AES_data/')
import classify_2HL 

def runConfig(dataDir, trainDataSize, resultDir, modelName, hidden1Size, hidden2Size, batchSize):
	trainData, devData, testData = classify_2HL.getData(dataDir, trainDataSize)
	
	x_train, y_train_oh = trainData
	x_dev, y_dev_oh = devData
	x_test, y_test_oh = testData
	
	## Instantiate the model and test, dev and training sets
	classifier = classify_2HL.Classifier(resultDir, modelName, x_train, y_train_oh, x_dev, y_dev_oh, x_test, y_test_oh, hidden1Size, hidden2Size)
	
	## Train the model
	classifier.train(batchSize)
	
	## Evaluate
	classifier.evaluate()
	
	##Save the model
	classifier.saveModel()


dataDir = "/xdisk/manojgopale/data_csv/config5p4/"
resultDir = "/extra/manojgopale/AES_data/config5p4_15ktraining/result_new"
trainDataSize = 15000

## 100, 100 neurons
modelName = "model_2hl_100_100neurons"
hidden1Size = 100
hidden2Size = 100
batchSize = 2048

runConfig(dataDir, trainDataSize, resultDir, modelName, hidden1Size, hidden2Size, batchSize)

###-------------###

## 500, 500 neurons
modelName = "model_2hl_500_500neurons"
hidden1Size = 500
hidden2Size = 500
batchSize = 2048

runConfig(dataDir, trainDataSize, resultDir, modelName, hidden1Size, hidden2Size, batchSize)

###-------------###

## 1000, 50 neurons
modelName = "model_2hl_1000_50neurons"
hidden1Size = 1000
hidden2Size = 50
batchSize = 2048

runConfig(dataDir, trainDataSize, resultDir, modelName, hidden1Size, hidden2Size, batchSize)

###-------------###
