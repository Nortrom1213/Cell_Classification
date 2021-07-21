import pandas as pd
from sklearn.naive_bayes import GaussianNB
import pickle

print("\n\nStarting.\n")
trainD = pd.HDFStore('train_data.h5')
testD = pd.HDFStore('test_data.h5')

# Splitting Data
dataX = trainD['rpkm']
dataY = trainD['labels']

testX = testD['rpkm']
testY = testD['labels']
print("Data splitting finished. \n")

# Fit Data (train)
gnb = GaussianNB()
gnb.fit(dataX, dataY)
print("Train finished.\n")

# Prediction
gnb_predict = gnb.predict(testX)
print("Prediction finished.\n Result: \n", gnb_predict)

# Error Calculation
count = 0
total = 0

for i, j in zip(gnb_predict, testY):
    total += 1
    if i == j:
        count += 1

print("Total length: ", total)
print('Finished count: ', count)
print("Accuracy: ", float(count / total))
#Accuracy:  0.4266199649737303