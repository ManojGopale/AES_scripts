import sys
sys.path.insert(0, '/extra/manojgopale/AES_data/')
import classify 

dataDir = "/xdisk/manojgopale/data_csv/config5p4/"
trainData, devData, testData = classify.getData(dataDir, 15000)

x_train, y_train_oh = trainData
x_dev, y_dev_oh = devData
x_test, y_test_oh = testData

## Instantiate the model and test, dev and training sets
resultDir = "/extra/manojgopale/AES_data/config5p4_15ktraining/result_new"
modelName = "m_newscript"

classifier = classify.Classifier(resultDir, modelName, x_train, y_train_oh, x_dev, y_dev_oh, x_test, y_test_oh)

## Train the model
classifier.train(2048)

## Evaluate
classifier.evaluate()

##Save the model
classifier.saveModel()
