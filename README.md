# TensorFlowMnist

基于TensorFlow 1.0.1实现的[MNIST手写数字识别](http://yann.lecun.com/exdb/mnist/)

## 数据集
所有的数据集都在data目录下：
- train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 
- train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 
- t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 
- t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)

## 网络模型的创建与训练
```train_model.py```： 创建并训练网络模型

## 网络模型的测试
```test_model.py```: 测试训练好后的网络模型

## 读取数据集
``` input_data.py```和```mnist_loader.py```

## 小工具：保存MNIST中的每一张图像
```extract_mnist.py```