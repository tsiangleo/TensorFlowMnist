#!/usr/bin/env python
#coding: utf-8

import tensorflow as tf
import input_data
from tensorflow.python.framework import graph_util


def build_network(height,width):
    """
    Function：构建网络模型。
    
    Parameters
    ----------
        height: Mnist图像的宽。
        width: Mnist图像的宽。
    
    """
    
    x = tf.placeholder(tf.float32, [None, height, width], name='input')
    
    y_placeholder = tf.placeholder(tf.float32, shape=[None, 10],name='labels_placeholder')
    
    keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
  
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  
    x_image = tf.reshape(x, [-1,height, width,1])
  
    # First Convolutional Layer
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    # Second Convolutional Layer
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    # Densely Connected Layer
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob_placeholder)

    # Readout Layer
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
    sofmax_out = tf.nn.softmax(logits,name="out_softmax")
   
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_placeholder))
    
    optimize = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    
    prediction_labels = tf.argmax(sofmax_out, axis=1,name="output")
    real_labels= tf.argmax(y_placeholder, axis=1)
    
    correct_prediction = tf.equal(prediction_labels, real_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #一个Batch中预测正确的次数
    correct_times_in_batch = tf.reduce_sum(tf.cast(correct_prediction, tf.int32))
    
    return dict(
                keep_prob_placeholder = keep_prob_placeholder,
                x_placeholder= x,
                y_placeholder = y_placeholder,
                optimize = optimize,
                logits = logits,
                prediction_labels = prediction_labels,
                real_labels = real_labels,
                correct_prediction = correct_prediction,
                correct_times_in_batch = correct_times_in_batch,
                cost = cost,
                accuracy = accuracy,
    )

def train_network(graph,
                 dataset,
                 batch_size,
                 num_epochs,
                 pb_file_path,):
    """
    Function：训练网络。
    
    Parameters
    ----------
        graph: 一个dict,build_network函数的返回值。
        dataset: 数据集
        batch_size: 
        num_epochs: 训练轮数。
        pb_file_path：要生成的pb文件的存放路径。
    """
    
    with tf.Session() as sess:
    
        sess.run(tf.global_variables_initializer())
         
        print "batch size:",batch_size
         
        #用于控制每epoch_delta轮在train set和test set上计算一下accuracy和cost
        epoch_delta = 2
        for epoch_index in range(num_epochs):
                    
            #################################
            #    获取TRAIN set，开始训练网络
            #################################
            for (batch_xs,batch_ys) in dataset.train.mini_batches(batch_size):
                sess.run([graph['optimize']], feed_dict={
                    graph['x_placeholder']: batch_xs,
                    graph['y_placeholder']: batch_ys,
                    graph['keep_prob_placeholder']:0.5,
                })
      
            
            #每epoch_delta轮在train set和test set上计算一下accuracy和cost
            if epoch_index % epoch_delta  == 0:
                #################################
                #    开始在 train set上计算一下accuracy和cost
                #################################
                #记录训练集中有多少个batch
                total_batches_in_train_set = 0
                #记录在训练集中预测正确的次数
                total_correct_times_in_train_set = 0
                #记录在训练集中的总cost
                total_cost_in_train_set = 0.
                for (train_batch_xs,train_batch_ys) in dataset.train.mini_batches(batch_size):
                    return_correct_times_in_batch = sess.run(graph['correct_times_in_batch'], feed_dict={
                        graph['x_placeholder']: train_batch_xs,
                        graph['y_placeholder']: train_batch_ys,
                        graph['keep_prob_placeholder']:1.0,
                    })
                    mean_cost_in_batch = sess.run(graph['cost'], feed_dict={
                        graph['x_placeholder']: train_batch_xs,
                        graph['y_placeholder']: train_batch_ys,
                        graph['keep_prob_placeholder']:1.0,
                    })
                      
                    total_batches_in_train_set += 1
                    total_correct_times_in_train_set += return_correct_times_in_batch
                    total_cost_in_train_set  += (mean_cost_in_batch*batch_size)
                      
            
                #################################
                # 开始在 test set上计算一下accuracy和cost
                #################################
                #记录测试集中有多少个batch
                total_batches_in_test_set = 0
                #记录在测试集中预测正确的次数
                total_correct_times_in_test_set = 0
                #记录在测试集中的总cost
                total_cost_in_test_set = 0.
                for (test_batch_xs,test_batch_ys) in dataset.test.mini_batches(batch_size):
                    return_correct_times_in_batch = sess.run(graph['correct_times_in_batch'], feed_dict={
                        graph['x_placeholder']: test_batch_xs,
                        graph['y_placeholder']: test_batch_ys,
                        graph['keep_prob_placeholder']:1.0,
                    })
                    mean_cost_in_batch = sess.run(graph['cost'], feed_dict={
                        graph['x_placeholder']: test_batch_xs,
                        graph['y_placeholder']: test_batch_ys,
                        graph['keep_prob_placeholder']:1.0,
                    })
                      
                    total_batches_in_test_set += 1
                    total_correct_times_in_test_set += return_correct_times_in_batch
                    total_cost_in_test_set  += (mean_cost_in_batch*batch_size)
                 
                ### summary and print
                acy_on_test = total_correct_times_in_test_set / float(total_batches_in_test_set * batch_size)
                acy_on_train = total_correct_times_in_train_set / float(total_batches_in_train_set * batch_size)
                print('Epoch - {:2d} , acy_on_test:{:6.2f}%({}/{}),loss_on_test:{:6.2f}, acy_on_train:{:6.2f}%({}/{}),loss_on_train:{:6.2f}'.
                      format(epoch_index, acy_on_test*100.0,total_correct_times_in_test_set,
                             total_batches_in_test_set * batch_size,total_cost_in_test_set, acy_on_train*100.0,
                             total_correct_times_in_train_set,total_batches_in_train_set * batch_size,total_cost_in_train_set))    
            
            # 每轮训练完后就保存为pb文件
            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output"]) #out_softmax
            with tf.gfile.FastGFile(pb_file_path,mode='wb') as f:
                f.write(constant_graph.SerializeToString())
                
                

def main():
    
    batch_size = 20
    num_epochs = 50
    
    #pb文件保存路径
    pb_file_path = "output/mnist-tf1.0.1.pb"

    g = build_network(height=28, width=28)
    dataset = input_data.read_data_sets()
    train_network(g, dataset, batch_size, num_epochs, pb_file_path)
    
main()