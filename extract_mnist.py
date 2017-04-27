#!/usr/bin/env python
#coding: utf-8

"""
MNIST数据集的原始格式是gz格式的，该脚本的功能是将该数据集中的每一张图片转为.png格式，并保存到指定目录。
"""

import gzip
import os
import numpy
import PIL.Image as Image


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_and_save_mnist(file_path,save_dir):
  """
    file_path:mnist数据集的路径，比如"./data/t10k-images-idx3-ubyte.gz"
    save_dir:要保存至的目标目录，比如"/home/u2/data"
  """
  f=open(file_path, 'rb')
  print('Extracting', f.name)
  
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols)
    
    
    #依次保存图片
    for index,d in enumerate(data):
        imges = Image.fromarray(d).convert('L')
        abs_path = os.path.join(os.path.abspath(save_dir), str(index)+".png")
        imges.save(abs_path,'png')
    

def main():
    TEST_IMAGES_PATH = 'data/t10k-images-idx3-ubyte.gz'
    extract_and_save_mnist(TEST_IMAGES_PATH,"./img")
    
# main()    