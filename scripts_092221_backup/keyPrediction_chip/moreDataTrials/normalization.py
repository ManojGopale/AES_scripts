## After loading the data
import pandas as pd

mean = trainData[0].mean(axis=0)
std = trainData[0].std(axis=0)

train_df = pd.DataFrame(data=trainData[0])
new_train = train_df.apply(lambda x: (x-mean)/std, axis=1)
