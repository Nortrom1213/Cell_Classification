import pandas as pd
from sklearn.linear_model import SGDClassifier

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
clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)
clf.fit(dataX, dataY)
print("Train finished.\n")

# Prediction
sgd_predict = clf.predict(testX)
print("Prediction finished.\n Result: \n", sgd_predict)

# Error Calculation
count = 0
total = 0

for i, j in zip(sgd_predict, testY):
    total += 1
    if i == j:
        count += 1

print("Total length: ", total)
print('Finished count: ', count)
print("Accuracy: ", float(count / total))

trainD.close()
testD.close()

#Accuracy:  0.11943957968476357