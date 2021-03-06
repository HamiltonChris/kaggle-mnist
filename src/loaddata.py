# loaddata.py
# loads data from mnist data set and puts it into a vectorized form
import pandas as pd
import numpy as np

def load_train_data():
    mnist = pd.read_csv("../data/train.csv")
    labels = one_hot_encoding(mnist["label"].values)
    train_data = mnist.drop("label",1).values
    train_data = train_data * (1/255)
    return train_data.T, labels

def one_hot_encoding(labels):
    labels = np.array(labels, dtype=np.uint)
    shape = labels.shape
    min_val = np.amin(labels)
    max_val = np.amax(labels) - min_val
    encoding = np.zeros([shape[0],max_val + 1])
    for i in range(shape[0]):
        encoding[i,labels[i]] = 1
    
    return encoding


def load_test_data():
    test_data = pandas.read_csv("../data/test.csv")
    test_data = test_data * (1/255)

    return test_data.T

#pad the data along the column 
def pad_data(data):
    length, width = data.shape
    pad = np.ones((length,1))
    return np.concatenate((pad,data),axis=1)
