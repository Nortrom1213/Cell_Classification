import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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
clf = RandomForestClassifier(max_depth=5, random_state=1)
clf.fit(dataX, dataY)
print("Train finished.\n")

# Prediction
rf_predict = clf.predict(testX)
print("Prediction finished.\n Result: \n", rf_predict)

# Error Calculation
count = 0
total = 0

for i, j in zip(rf_predict, testY):
    total += 1
    if i == j:
        count += 1

print("Total length: ", total)
print('Finished count: ', count)
print("Accuracy: ", float(count / total))
#Accuracy:  0.3828371278458844