import pandas as pd
from sklearn.naive_bayes import GaussianNB

print("\n\nStarting.\n")
trainD = pd.HDFStore('train_pca.h5')
testD = pd.HDFStore('test_pca.h5')

# Splitting Data
dataX = trainD['data']
dataY = trainD['labels']

testX = testD['data']
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
#Accuracy:  0.30122591943957966 FOR N =100
#Accuracy:  0.30017513134851137 FOR N =200