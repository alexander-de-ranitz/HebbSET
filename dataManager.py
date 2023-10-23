import numpy as np
import scipy.io as sio
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.datasets import cifar100
from keras.datasets import cifar10

class DataManager:
    @staticmethod
    def get_lung_data():
        # load data
        mat = sio.loadmat('data/lung.mat') #lung dataset was downloaded from http://featureselection.asu.edu/
        # As the lung dataset has just few hundred samples, and few thousands features, you will observe a high variance in the accuracy from one epoch to another.
        # We chose this dataset to show how the SET-MLP model handles overfitting.
        # To see a much more stable behaviour of the model please experiment with datasets with a higher amount of samples, e.g. COIL-100 can be a really nice one as it has 100 classes and also the last layer will be sparse.
        X = mat['X']
        
        # one hot encoding
        noClasses = np.max(mat['Y'])
        Y=np.zeros((mat['Y'].shape[0],noClasses))
        for i in range(Y.shape[0]):
            Y[i,mat['Y'][i]-1]=1

        #split data in training and testing
        indices=np.arange(X.shape[0])
        np.random.shuffle(indices)
        X_train=X[indices[0:int(X.shape[0]*3/4)]]
        Y_train=Y[indices[0:int(X.shape[0]*3/4)]]
        X_test=X[indices[int(X.shape[0]*3/4):]]
        Y_test=Y[indices[int(X.shape[0]*3/4):]]

        # Normalise data to [0,1]
        X_train = X_train.astype('float64').transpose()
        X_test = X_test.astype('float64').transpose()

        X_TrainMin = np.min(X_train, axis=0)
        X_train -= X_TrainMin
        X_TrainMax = np.max(X_train, axis=0)
        X_train /= X_TrainMax + 0.0001
        X_TestMin = np.min(X_test, axis=0)
        X_test -= X_TestMin
        X_TestMax = np.max(X_test, axis=0)
        X_test /= X_TestMax + 0.0001

        X_train = X_train.transpose()
        X_test = X_test.transpose()
        
        return (X_train, Y_train), (X_test, Y_test)

    def get_MNIST_data(percentage_to_use = 1.0):
        # Get MNIST data
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data(path="mnist.npz")

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
        Y_train = to_categorical(Y_train)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
        Y_test = to_categorical(Y_test)
        
        # Shuffle data
        indices_train =np.arange(X_train.shape[0])
        indices_test =np.arange(X_test.shape[0])
        np.random.shuffle(indices_train)
        np.random.shuffle(indices_test)
        X_train= X_train[indices_train]
        Y_train= Y_train[indices_train]
        X_test= X_test[indices_test]
        Y_test= Y_test[indices_test]
        
        X_train = X_train.astype('float64').transpose()
        X_test = X_test.astype('float64').transpose()

        X_TrainMin = np.min(X_train, axis=0)
        X_train -= X_TrainMin
        X_TrainMax = np.max(X_train, axis=0)
        X_train /= X_TrainMax + 0.0001
        X_TestMin = np.min(X_test, axis=0)
        X_test -= X_TestMin
        X_TestMax = np.max(X_test, axis=0)
        X_test /= X_TestMax + 0.0001

        X_train = X_train.transpose()
        X_test = X_test.transpose()
        

        X_train = X_train[:int(percentage_to_use * len(X_train))]
        Y_train = Y_train[:int(percentage_to_use * len(Y_train))]
        X_test = X_test[:int(percentage_to_use * len(X_test))]
        Y_test = Y_test[:int(percentage_to_use * len(Y_test))]

        return (X_train, Y_train), (X_test, Y_test)
    
    @staticmethod
    def get_CIFAR_100_data():
        (X_train, Y_train), (X_test, Y_test) = cifar100.load_data(label_mode="coarse")

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3])
        Y_train = to_categorical(Y_train)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2] * X_test.shape[3])
        Y_test = to_categorical(Y_test)
        
        indices_train = np.arange(X_train.shape[0])
        indices_test = np.arange(X_test.shape[0])
        np.random.shuffle(indices_train)
        np.random.shuffle(indices_test)
        X_train= X_train[indices_train]
        Y_train= Y_train[indices_train]
        X_test= X_test[indices_test]
        Y_test= Y_test[indices_test]
        
        X_train = X_train.astype('float64').transpose()
        X_test = X_test.astype('float64').transpose()

        X_TrainMin = np.min(X_train, axis=0)
        X_train -= X_TrainMin
        X_TrainMax = np.max(X_train, axis=0)
        X_train /= X_TrainMax + 0.0001
        X_TestMin = np.min(X_test, axis=0)
        X_test -= X_TestMin
        X_TestMax = np.max(X_test, axis=0)
        X_test /= X_TestMax + 0.0001

        X_train = X_train.transpose()
        X_test = X_test.transpose()
        
        X_train = X_train[:5000]
        Y_train = Y_train[:5000]
        X_test = X_test[:500]
        Y_test = Y_test[:500]

        return (X_train, Y_train), (X_test, Y_test)
    
    @staticmethod
    def get_CIFAR_10_data():
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2] * X_train.shape[3])
        Y_train = to_categorical(Y_train)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2] * X_test.shape[3])
        Y_test = to_categorical(Y_test)
        
        indices_train = np.arange(X_train.shape[0])
        indices_test = np.arange(X_test.shape[0])
        np.random.shuffle(indices_train)
        np.random.shuffle(indices_test)
        X_train= X_train[indices_train]
        Y_train= Y_train[indices_train]
        X_test= X_test[indices_test]
        Y_test= Y_test[indices_test]
        
        X_train = X_train.astype('float64').transpose()
        X_test = X_test.astype('float64').transpose()

        X_train /= 255.0
        X_test /= 255.0
        # X_TrainMin = np.min(X_train, axis=0)
        # X_train -= X_TrainMin
        # X_TrainMax = np.max(X_train, axis=0)
        # X_train /= X_TrainMax + 0.0001
        # X_TestMin = np.min(X_test, axis=0)
        # X_test -= X_TestMin
        # X_TestMax = np.max(X_test, axis=0)
        # X_test /= X_TestMax + 0.0001

        X_train = X_train.transpose()
        X_test = X_test.transpose()
        
        X_train = X_train
        Y_train = Y_train
        X_test = X_test
        Y_test = Y_test

        return (X_train, Y_train), (X_test, Y_test)