import pandas as pd
store=pd.HDFStore('test_pca.h5')
labels = []
for i in store['labels']:
    if i not in labels:
        labels.append(i)
for i in labels:
    print("'" + i + "',")
store.close()