import functools
from matplotlib import pyplot as plt
import numpy as np
import random as rnd

dataset = np.load('/mnt/c/Users/andre/Projects/AN2DL_23-24_Apple_Elixir/res/public_data_clean.npz', allow_pickle=True)
keys = list(dataset.keys())
images = np.array(dataset[keys[0]])
labels = np.array(dataset[keys[1]])


# this method oversample the class with less elements (we assume that is the negative class)
def balance_sets_add(images,labels):
    negative_n = functools.reduce(lambda a, b: a+b, labels)
    print(negative_n)
    positive_n = len(images) - negative_n
    print(positive_n)
    to_add = positive_n - negative_n
    X_n = images[labels == 1]
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


# this method oversample the class with less elements (assumed as the negative class)
# and undersample the class with more elements (respectively the positive class)
def balance_sets(images,labels):
    negative_n = functools.reduce(lambda a, b: a+b, labels)
    print(negative_n)
    positive_n = len(images) - negative_n
    print(positive_n)
    to_add = int((positive_n - negative_n)/2)
    to_delete = to_add
    X_n = images[labels == 1]
    deletable = []
    for i in range(0,to_delete):
      to_delete -= 1
      el = rnd.randrange(0,len(labels))
      while(labels[el] == 1 or el in deletable):
        el = rnd.randrange(0,len(labels))
      deletable.append(el)

    images = np.delete(images, deletable, axis=0)
    labels = np.delete(labels, deletable, axis=0)  

    for i in range(0,to_add):
      to_add -= 1
      el = rnd.randrange(0,negative_n - 1)
      images = np.insert(images,0,X_n[el],axis=0)
      labels = np.insert(labels,0,1,axis=0)
      if to_add <= 0:
         break
  
    print(functools.reduce(lambda a, b: a+b, labels))
    print(len(images) - negative_n)
    return images,labels

images, labels = balance_sets(images,labels)

np.savez("/mnt/c/Users/andre/Projects/AN2DL_23-24_Apple_Elixir/res/public_data_balanced.npz", images, labels)