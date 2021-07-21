import pandas as pd
from sklearn import tree

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
clf = tree.DecisionTreeClassifier(max_depth=50)
clf.fit(dataX, dataY)
print("Train finished.\n")

# Prediction
dt_predict = clf.predict(testX)
print("Prediction finished.\n Result: \n", dt_predict)

# Error Calculation
count = 0
total = 0

for i, j in zip(dt_predict, testY):
    total += 1
    if i == j:
        count += 1

print("Total length: ", total)
print('Finished count: ', count)
print("Accuracy: ", float(count / total))

trainD.close()
testD.close()

#Accuracy:  0.22942206654991243