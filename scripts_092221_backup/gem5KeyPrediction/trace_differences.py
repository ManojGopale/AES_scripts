import numpy as np
import pandas as pd
from optparse import OptionParser

def get_comma_separated_args(option, opt, value, parser):
	print("Inside callback\n")
	setattr(parser.values, option.dest, value.split(','))

parser = OptionParser()

########-----------------------------------------------------#########
parser.add_option('--numConfigs',
									action = 'store', type='int', dest='numConfigs', default = 1)
parser.add_option('--baselineList',
                  type='string',
									action='callback',
									callback=get_comma_separated_args,
									dest = "getBaselineList")

parser.add_option('--targetList',
                  type='string',
									action='callback',
									callback=get_comma_separated_args,
									dest = "getTargetList")

(options, args) = parser.parse_args()

baselineList = options.getBaselineList
targetList = options.getTargetList ##For Date paper, targetList is hardcoded below.
numConfigs = options.numConfigs

########-----------------------------------------------------#########
## I think the baseline should be picked from Train, since the model is trained on it.
## The target should be taken from test as the test will be the one that the model will be tested on.
## tracesPerConfig -> Traces to pick from a specific key per baselineconfiguration. This will change from 
## tracesPerConfig=7500 for 1 config, 7500/2 for 2 configs, 7500/3 for 3 configs
def traceDifference(baselineConfig, targetConfig, tracesPerConfig):
	## Get the baseline data for key=15(chosen at random) to compare against mean from other configsconfig
	data_baseline = pd.DataFrame()
	for config in baselineConfig:
		baselineCsvPath = "/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/" + config + "/aesData_config9_Train_0.csv"
		data_baseline_inter = pd.read_csv(baselineCsvPath, header=None)
		## Get key==15 traces, shuffle them, and take top traces for concatenation
		data_baseline_inter = data_baseline_inter.loc[data_baseline_inter.iloc[:,1500]==15].sample(frac=1).reset_index(drop=True).iloc[0:tracesPerConfig,0:1500]
		data_baseline = pd.concat([data_baseline, data_baseline_inter], ignore_index=True)
	key15_baseline_train_mean = data_baseline.mean(axis=0)
	key15_baseline_train_std =  data_baseline.std(axis=0)
	
	## Get mean for the same key(15) on a target configuration and see id mean lies in the 3sigma limit
	targetCsvPath = "/xdisk/rlysecky/manojgopale/xdisk/gem5DataCollection/csvResult/" + targetConfig + "/aesData_config9_Test_0.csv"
	data_target = pd.read_csv(targetCsvPath, header=None)
	key15_target_500 = data_target.loc[data_target.iloc[:,1500]==15].sample(frac=1).reset_index(drop=True).iloc[0:500,0:1500]
	key15_target_test_mean = key15_target_500.mean(axis=0)
	key15_target_test_std =  key15_target_500.std(axis=0)
	print("data_baseline shape = %s, data_target shape= %s" %(data_baseline.shape, key15_target_500.shape))
	
	
	## Analysis
	## Get the count where the target is out of bounds
	(np.abs(key15_baseline_train_mean-key15_target_test_mean) > 3*key15_baseline_train_std).value_counts()
	#False    1353
	#True      148
	## To directly get the count
	count_outside = key15_baseline_train_mean[np.abs(key15_baseline_train_mean-key15_target_test_mean) > 3*key15_baseline_train_std].count()
	print("baseline config= %s target config= %s difference in count= %s\n" %(baselineConfig, targetConfig, count_outside))
	
	## this gets the index where the keys are putside the limits
	key15_baseline_train_mean[np.abs(key15_baseline_train_mean-key15_target_test_mean) > 3*key15_baseline_train_std].index.values


targetList= ["config3p1", "config3p2", "config3p3", "config3p4", "config4p1", "config4p2", "config4p3", "config4p4","config5p1", "config5p2", "config5p3", "config5p4"]
## 2500x3=7500 traces/key was found to be optimal training size. So to keep the training size identical for 2 and 1 configs, 
## we need to make sure that the trainig size doesn't wary
tracesPerConfig = int(7500/numConfigs)
##M
##MbaselineConfig = ["config3p3", "config3p4", "config4p3", "config4p4", "config5p3", "config5p4"]
#for idx, baselineConfig in enumerate(baseList):
for index, targetConfig in enumerate(targetList):
	traceDifference(baselineList, targetConfig, tracesPerConfig)


#python3.5 /xdisk/rlysecky/manojgopale/extra/gem5KeyPrediction/scr/trace_differences.py > ~/GEM5_Analysis/trace_difference_group_output_allcombo.log
