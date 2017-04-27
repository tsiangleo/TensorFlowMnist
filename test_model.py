#!/usr/bin/env python
#coding: utf-8

import tensorflow as tf
import  numpy as np
import PIL.Image as Image
            

def recognize(png_path,pb_file_path):
    """
    Function：使用训练完的网络模型进行预测。
    
    Parameters
    ----------
        png_path：要预测的图片的路径。
        pb_file_path: pb文件的路径。
    
    """    
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
    
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read()) #rb
            _ = tf.import_graph_def(output_graph_def, name="")
    
        with tf.Session() as sess:
            tf.global_variables_initializer().run()

            input_x = sess.graph.get_tensor_by_name("input:0")
            print input_x
            out_softmax = sess.graph.get_tensor_by_name("out_softmax:0")
            print out_softmax
            keep_prob = sess.graph.get_tensor_by_name("keep_prob_placeholder:0")
            print keep_prob
            out_label = sess.graph.get_tensor_by_name("output:0")
            print out_label
             
            img_datas  = np.array(Image.open(png_path).convert('L')) 
            img_out_softmax = sess.run(out_softmax, feed_dict={
                                   input_x: img_datas,
                                   keep_prob: 1.0,
                               })
             
            print "img_out_softmax:",img_out_softmax
            prediction_labels = np.argmax(img_out_softmax, axis=1)
            print "label:",prediction_labels
            
          
#recognize("/home/tsiangleo/mnist_test_set/1.png","output/mnist-tf1.0.1.pb")

