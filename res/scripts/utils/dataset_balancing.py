import functools
from matplotlib import pyplot as plt
import numpy as np
import random as rnd

dataset = np.load('/mnt/c/Users/andre/Projects/AN2DL_23-24_Apple_Elixir/res/public_data_clean.npz', allow_pickle=True)
keys = list(dataset.keys())
images = np.array(dataset[keys[0]])
labels = np.array(dataset[keys[1]])


negative_n = functools.reduce(lambda a, b: a+b, labels)
print(negative_n)
positive_n = len(images) - negative_n
print(positive_n)

def balance_sets(images,labels):
    negative_n = functools.reduce(lambda a, b: a+b, labels)
    print(negative_n)
    positive_n = len(images) - negative_n
    print(positive_n)
    to_add = positive_n - negative_n
    X_n = images[labels == 1]
    y_n = labels[labels == 1]
    for i in range(0,to_add):
      to_add -= 1
      el  = rnd.randrange(0,negative_n - 1)
      images = np.insert(images,0,X_n[el],axis=0)
      labels = np.insert(labels,0,1,axis=0)
      print(i)
      if to_add <= 0:
         break
    negative_n = functools.reduce(lambda a, b: a+b, labels)
    assert (len(labels) - negative_n * 2)==  0
    return images,labels

images, labels = balance_sets(images,labels)


np.savez("/mnt/c/Users/andre/Projects/AN2DL_23-24_Apple_Elixir/res/public_data_balanced.npz", images, labels)