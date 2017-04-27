#!/usr/bin/env python
#coding: utf-8

import numpy as np
import mnist_loader
import collections


Datasets = collections.namedtuple('Datasets', ['train',  'test'])

class DataSet(object):
 
  def __init__(self,
               images,
               labels):

    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
 
  @property
  def images(self):
    return self._images
 
  @property
  def labels(self):
    return self._labels
 
  @property
  def num_examples(self):
    return self._num_examples
 
  @property
  def epochs_completed(self):
    return self._epochs_completed
  

  def mini_batches(self,mini_batch_size):
    """
      return: list of tuple(x,y)
    """
    # Shuffle the data
    perm = np.arange(self._num_examples)
    np.random.shuffle(perm)
    self._images = self._images[perm]
    self._labels = self._labels[perm]
    
    n = self.images.shape[0]
    
    mini_batches = [(self._images[k:k+mini_batch_size],self._labels[k:k+mini_batch_size])
                    for k in xrange(0, n, mini_batch_size)]
    
    if len(mini_batches[-1]) != mini_batch_size:
        return mini_batches[:-1]
    else:
        return mini_batches


  def _next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
        # Finished epoch
        self._epochs_completed += 1
        # Shuffle the data
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self._images[perm]
        self._labels = self._labels[perm]
        # Start next epoch
        start = 0
        self._index_in_epoch = batch_size
        assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_data_sets():
    """
    Function：读取训练集(TrainSet)和测试集(TestSet)。
    
    Notes
    ----------
        TrainSet: include imgs_train and labels_train.
        TestSet:  include imgs_test and  labels_test.
        
        the shape of imgs_train and imgs_test are:(batch_size,height,width). namely (n, 28L, 28L)
        the shape of labels_train and labels_test are:(batch_size,num_classes). namely (n, 10L)
    
    """
    imgs_train, imgs_test, labels_train, labels_test  =  mnist_loader.read_data_sets()
    train = DataSet(imgs_train, labels_train)
    test = DataSet(imgs_test, labels_test)
    return Datasets(train=train, test=test)

def _test():
    dataset = read_data_sets()
    
    print "dataset.train.images.shape:",dataset.train.images.shape
    print "dataset.train.labels.shape:",dataset.train.labels.shape
    print "dataset.test.images.shape:",dataset.test.images.shape
    print "dataset.test.labels.shape:",dataset.test.labels.shape
    
    print dataset.test.images[0]
    print dataset.test.labels[0]

# _test()