import sys
sys.path.insert(0, '/xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/scr/')
import getMaxTracesperKey
import time
import numpy as np

from optparse import OptionParser

parser = OptionParser()
parser.add_option('--csvPath',
									action = 'store', type='string', dest='csvPath', default = '/xdisk/rlysecky/manojgopale/extra/keyPrediction_chip/scr/moreDataTrials/result/rerun_177_trainSize_15000_5_032121_4HL_169epochs_54p07_acc_outputPredict.csv')
parser.add_option('--interval',
									action = 'store', type='int', dest='interval', default = 10)

(options, args) = parser.parse_args()

########
## csvPath is kept as deault for now, can be canged later on
csvPath = options.csvPath
interval = options.interval

getMaxTracesperKey.getMaxTracesperKey(csvPath, interval)
