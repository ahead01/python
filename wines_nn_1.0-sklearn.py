# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 20:37:08 2018

@author: Austin
"""
import numpy as np
import matplotlib.pyplot as plt


DESCR="""Red-Wine Quality Data Set  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236."""

labels=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
        'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol','quality']

reds=r"C:\Users\Austin\Towson\DataMining\Assingnments\Term_Project\DataSets\wine\winequality-white-processed.npy"
data=np.load(reds)
print(str(type(data)))
print('Shape: ' + str(np.shape(data)))
print('Size: ' + str(np.size(data)))

# Perform some pre-processing
plt.hist(data[:,11], bins=10)
plt.title("Histogram of labels")
plt.show()

# trash outliers with the median
#d = np.abs(data[:,11] - np.median(data[:,11]))
#mdev = np.median(d)
#s = d/mdev if mdev else 0.
#trim=np.where(data[:,11] < 2*s)
#data=np.delete(data, trim, axis=0 )

# Perform some pre-processing
plt.hist(data[:,11], bins=10)
plt.title("Histogram of labels")
#plt.show()

zeros=np.zeros(np.shape(data)[0])
target=data[:,11]
data=np.delete(data,  [11], axis=1 )

#data=np.delete(data,  [1,5,6,7,8,9,10,11], axis=1 )
#data=np.delete(data, [2,3,4,11], axis=1)
#data=np.delete(data,  [1,2,3,4,5,6,7,8,9,11], axis=1 )
#data=np.delete(data,  [1,2,3,4,5,6,7,9,11], axis=1 )
#data=np.delete(data,  [1,2,3,4,5,6,9,11], axis=1 )
# split the data to train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target,test_size=.25)

#scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
# Fit only to the training data
scaler.fit(X_train)

# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Now train
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(200,200),
                    activation='relu', alpha=0.01, 
                    batch_size='auto', beta_1=0.9, 
                    learning_rate='constant', learning_rate_init=0.001, max_iter=1000, momentum=0.9,
                    nesterovs_momentum=True, power_t=0.5, random_state=None,
                    shuffle=True, solver='adam', tol=1, validation_fraction=0.1,
                    verbose=False, warm_start=False, beta_2=0.999, early_stopping=False, epsilon=1e-08)
mlp.fit(X_train,y_train)


# npw predict
predictions = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

# coefs_ is a list of weight matrices, where weight matrix at index i represents the weights between layer i and layer i+1.
#print(str(mlp.coefs_[0]))
#print(str(len(mlp.coefs_[0])))
#print(str(len(mlp.intercepts_[0])))

# intercepts_ is a list of bias vectors, where the vector at index i represents the bias values added to layer i+1.