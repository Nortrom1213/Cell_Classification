import pandas as pd
from sklearn import svm

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
clf = svm.SVC()
clf.fit(dataX, dataY)
print("Train finished.\n")

# Prediction
svm_predict = clf.predict(testX)
print("Prediction finished.\n Result: \n", svm_predict)

# Error Calculation
count = 0
total = 0

for i, j in zip(svm_predict, testY):
    total += 1
    if i == j:
        count += 1

print("Total length: ", total)
print('Finished count: ', count)
print("Accuracy: ", float(count / total))

#Accuracy:  0.4921190893169877