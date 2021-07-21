import pandas as pd
from sklearn import neighbors
import pickle

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
max_weights = 'uniform'
max_n = 0
max_accuracy = 0
for weights in ['uniform', 'distance']:
    for n_neighbors in [8, 15, 30, 50, 100,500, 1000]:
        print("weight = ", weights, ", n_neighbors = ", n_neighbors)
        clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights, algorithm='kd_tree')
        clf.fit(dataX, dataY)
        print("Train finished.\n")

        # Prediction
        knn_predict = clf.predict(testX)
        print("Prediction finished.\n Result: \n", knn_predict)

        # Error Calculation
        count = 0
        total = 0

        for i, j in zip(knn_predict, testY):
            total += 1
            if i == j:
                count += 1

        if max_accuracy < float(count / total):
            max_weights = weights
            max_n = n_neighbors
            max_accuracy = float(count / total)

        print("Total length: ", total)
        print('Finished count: ', count)
        print("Accuracy: ", float(count / total))
        print("\n")

print("Max Weight: ", max_weights)
print("Max N: ", max_n)
print("Max Accuracy: ", max_accuracy)

trainD.close()
testD.close()

#Max Weight:  uniform
#Max N:  1000
#Max Accuracy:  0.45569176882662

