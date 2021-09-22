import numpy as np
import pandas as pd


## Collate mean's and std dev
#MconfigList = ["config3p1", "config3p2", "config3p3", "config3p4", "config4p1", "config4p2", "config4p3", "config4p4", "config5p1", "config5p2", "config5p3", "config5p4"]
configList = ["config3p1", "config3p2"]
## Sample lenght when the mean and std were calculated
## halfTraces is I think 28000/2=14000
sampleLength = 14000
cumm_mean = pd.DataFrame(np.zeros((1500,1)))
cumm_std = pd.DataFrame(np.zeros((1500,1)))
## Since each config has 2 samples of mean's the sample size will be 2*number of configs considered for pooling
numPoolSamples = len(configList)*2

for index, configName in enumerate(configList):
	## Load means and std Dev
	mean1Path = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/" + configName + "/mean1.csv"
	mean2Path = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/" + configName + "/mean2.csv"

	mean1 = pd.read_csv(mean1Path, header=None)
	mean2 = pd.read_csv(mean2Path, header=None)

	cumm_mean = cumm_mean + sampleLength*mean1
	cumm_mean = cumm_mean + sampleLength*mean2

	## Get the diffrences which are substanial
	## (mean1-mean2)[(mean1-mean2).abs()>0.0001].dropna()


	std1Path = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/" + configName + "/std1.csv"
	std2Path = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/" + configName + "/std2.csv"

	std1 = pd.read_csv(std1Path, header=None)
	std2 = pd.read_csv(std2Path, header=None)

	cumm_std = cumm_std + (sampleLength-1)*std1.pow(2)
	cumm_std = cumm_std + (sampleLength-1)*std2.pow(2)

## Consolidate the cummulatives to get final mean and std

mean_pool = cumm_mean/(sampleLength*numPoolSamples)
std_pool = (cumm_std/((sampleLength*numPoolSamples)-numPoolSamples)).pow(0.5)

## Saving the mean's and std dev to allConfigs folder
meanPoolPath = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/allConfigs/" + "config" + "_".join([i[-3:] for i in configList]) + "_mean.csv"
stdPoolPath = "/xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/result/allConfigs/" + "config" + "_".join([i[-3:] for i in configList]) + "_std.csv"
mean_pool.to_csv(meanPoolPath, index=False, header=False)
std_pool.to_csv(stdPoolPath, index=False, header=False)

print("\nMean1 diff\n")
print("%s" %((mean_pool-mean1)[(mean_pool-mean1).abs()>0.0001].dropna()))
print("\nMean2 diff\n")
print("%s" %((mean_pool-mean2)[(mean_pool-mean2).abs()>0.0001].dropna()))
print("\nstd1 diff\n")
print("%s" %((std_pool-std1)[(std_pool-std1).abs()>0.0001].dropna()))
print("\nstd2 diff\n")
print("%s" %((std_pool-std2)[(std_pool-std1).abs()>0.0001].dropna()))

