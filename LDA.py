import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
var_threshold=0.99
print("Start")
store = pd.HDFStore('all_data.h5')
rpkm_matrix = store['rpkm']
label = store['labels']
access = store['accessions']
store.close()
print(rpkm_matrix.shape)
print('Data Loading Done')
#Dimension Reduction:PCA
n_components = 2
LDA=LinearDiscriminantAnalysis(n_components=2)
LDA.fit(rpkm_matrix, label)
var=0
store=pd.HDFStore('train_data.h5')
train_data=store['rpkm']
train_pca=pd.DataFrame(LDA.transform(train_data)[:,:n_components])
print(train_pca.shape)
train_pca.to_hdf('train_lda.h5',key='data',mode='w')
store['labels'].to_hdf('train_lda.h5',key='labels')
store['accessions'].to_hdf('train_lda.h5',key='accessions')
store.close()
print("train transformed")
store=pd.HDFStore('test_data.h5')
test_data=store['rpkm']
test_pca=pd.DataFrame(LDA.transform(test_data)[:,:n_components])
print(test_pca.shape)
test_pca.to_hdf('test_lda.h5',key='data',mode='w')
store['labels'].to_hdf('test_lda.h5',key='labels')
store['accessions'].to_hdf('test_lda.h5',key='accessions')
store.close()
print("test transformed")
print('PCA Done')