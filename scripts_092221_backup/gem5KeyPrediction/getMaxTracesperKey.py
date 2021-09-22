import pandas as pd
import numpy as np

## csvPAth: Path of the csv path that stores the predicted key vs actual key for cw run
## key: actual key, for which we want to find the breakup
## interval: number of dev traces we want to consider
def getMaxTracesperKey(csvPath, interval):
	#df= pd.read_csv("/xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/result/rerun_177_4_031821_4HL_346epochs_58p77_acc_outputPredict.csv", header=None)
	df= pd.read_csv(csvPath ,header=None)
	print("csvFile= %s\n" %(csvPath))
	## Sort the first column in ascending order and also keep the order in which they appear in dev set
	## rename_axis names the index as "dfIndex"
	df_sorted = df.rename_axis("index").sort_values(by=[0,"index"], ascending=["True", "True"])
	
	#Mkey=12
	#Minterval=100
	for key in range(256):
		print("key= %s\n" %(key))
		df_inter = df_sorted.loc[df_sorted.iloc[:,0] == key].iloc[0:interval,]
		df_inter.columns = ['actualKey','predictedKey']
		error_df = df_inter[df_inter["actualKey"]!=df_inter["predictedKey"]].astype('category')
		
		error_df["predictPairs"] = error_df["actualKey"].astype(str).str.cat(error_df["predictedKey"].astype(str), sep="-")
		totalCount = df_inter["actualKey"].count()
		errorCount = error_df["predictPairs"].count()
		accuracy = ((df_inter["actualKey"].count()-error_df["predictPairs"].count())/df_inter["actualKey"].count())*100
		accuracy
		
		## Get the number of predicted values for all the keys predicted.
		## Then get the max predicted key and compare it to the actual key. If they match then report the percentage and 
		## that it was predicted most number of times
		## as_index will create a different column instead of using the column '"predictedKey"' as the index
		df_group=df_inter.groupby(by="predictedKey", as_index=False).count()
		df_group.columns=["predictedKey", "count"]
		
		maxPredictedKey = df_group.loc[df_group["count"].values.argmax()]["predictedKey"]
		maxPrecitedKeyCount = df_group.loc[df_group["count"].values.argmax()]["count"]
		maxPredictedKeyPercentage = (maxPrecitedKeyCount/interval)*100
		
		if (key == maxPredictedKey):
			print("actual key is %s, maxPredicted key is %s\nmaxPredictedCount= %s, maxPredictedKeyPercentage= %s, interval= %s\n" %(key, maxPredictedKey, maxPrecitedKeyCount, maxPredictedKeyPercentage, interval))
		else:
			print("###actual key is %s, maxPredicted key is %s\n###maxPredictedCount= %s, maxPredictedKeyPercentage= %s, interval= %s\n" %(key, maxPredictedKey, maxPrecitedKeyCount, maxPredictedKeyPercentage, interval))
