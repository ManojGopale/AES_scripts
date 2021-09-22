import numpy as np
import pandas as pd
from numpy import genfromtxt
from operator import itemgetter, attrgetter

csvPath = "/xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/scr/../result/rerun_177_4HL_195epochs_57p54_acc_outputPredict.csv"

## Load the csv file
data = genfromtxt(csvPath, delimiter=',')

## Get the sorted list, sorts the list by the 1st column and keeps the order intact 
## returns a list
sortData = sorted(data.tolist(), key=itemgetter(0))

## Get the top 'n' for each key
topN = [x for x in sortData[1000*key: 1000*key + n]]
print("size of topN= %s" %(topN.len))
