import numpy as np
import tensorflow as tf
import os
import sys
#from cwt import cwt
#from spec_aug import *
#from utils_aug import *

flags = tf.app.flags.FLAGS

#TODO:dropout, modeldepth

class Model:
	def build_encoder(self, inputs, params, name):
		self.dropout, self.istraining, self.batch_size = params
		with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
			layer = inputs
			width = [16,16] # list(map(int, flags.width.split('/')))
			
			params_init = list(map(int, flags.initializer.split('/')))
			self.initializer = tf.contrib.layers.variance_scaling_initializer(\
								factor=params_init[0], uniform=params_init[1])
			self.layers_encoded = []
			self.layers_encoded.append(layer)
			layer = self.conv_max_residual_stride_block(layer, width[0])
			self.layers_encoded.append(layer)
			
			channel = [64,64,64,128,128,128,128,256,256,256,256,512,512,512,512]
			subsample = [1,2,1,2,1,2,1,2,1,2,1,1,1,1,1]

			for i in range(len(channel)):
				layer = self.conv_max_residual_block(\
								layer, channel[i], 16, subsample[i], i)
				self.layers_encoded.append(layer)
			
			layer = tf.layers.batch_normalization(layer, training=self.istraining)
			layer = tf.nn.relu(layer)
			_, pool_size, _ = layer.get_shape().as_list()
			outputs = tf.layers.average_pooling1d(layer, pool_size, pool_size, 'same')
			return outputs
		
	
	def build_classifier(self, inputs, params, name, reuse=tf.AUTO_REUSE):
		self.dropout, self.istraining, self.batch_size = params
		with tf.variable_scope(name):
			outputs = self.dense_block_linear(inputs, flags.output_size, 'classifier')
			return outputs

	def dense_block_linear(self, inputs, output_size, name):
		istraining = self.istraining
		drop = self.dropout[2]
		layer = inputs
		with tf.variable_scope(name):
			layer = tf.reshape(layer, [flags.batch_size_gpu, -1])
			layer = tf.layers.dense(layer, output_size,	use_bias=True,
									kernel_initializer=self.initializer)
			logits = layer
			return logits
	
	def conv_max_residual_stride_block(self,inputs,kernel_size):
		istraining = self.istraining
		drop = self.dropout[0]
		with tf.variable_scope('res_block'):
			layer = inputs
			
			conv1 = tf.layers.conv1d(inputs=layer, 
									filters=64,
									kernel_size=kernel_size, 
									strides=1, 
									padding='same', 
									activation=None,
									use_bias=False,
									kernel_initializer=self.initializer)
			BN1 = tf.layers.batch_normalization(conv1, training=istraining)
			conv1_relu = tf.nn.relu(BN1)
			max_pool_1 = tf.layers.max_pooling1d(inputs=conv1_relu, 
													pool_size=2, 
													strides=2, 
													padding='same')
			conv2 = tf.layers.conv1d(inputs=conv1_relu, 
									filters=64,
									kernel_size=kernel_size, 
									strides=1, 
									padding='same', 
									activation=None,
									use_bias=False,
									kernel_initializer=self.initializer)
			conv2_BN2 = tf.layers.batch_normalization(conv2, training=istraining)
			conv2_BN2_relu = tf.nn.relu(conv2_BN2)
			conv2_BN2_relu_dropout = tf.layers.dropout(conv2_BN2_relu, drop, training=istraining)
			conv3 = tf.layers.conv1d(inputs=conv2_BN2_relu_dropout, 
									filters=64,
									kernel_size=kernel_size, 
									strides=2, 
									padding='same', 
									activation=None,
									use_bias=False,
									kernel_initializer=self.initializer)
			result = tf.add(max_pool_1,conv3)
			#print(result)
			return result 
	
	def conv_max_residual_block(self, inputs, channel, kernel_size, subsample_factor, i):
		istraining = self.istraining
		drop = self.dropout[1]
		with tf.variable_scope('res_block'+str(i)):
			BN_1 = tf.layers.batch_normalization(inputs, training=istraining)
			relu_1 = tf.nn.relu(BN_1)
			conv_1 = tf.layers.conv1d(inputs=relu_1, 
									filters=channel,
									kernel_size=kernel_size, 
									strides=subsample_factor, 
									padding='same', 
									activation=None,
									use_bias=False,
									kernel_initializer=self.initializer)
			BN_2 = tf.layers.batch_normalization(conv_1, training=istraining)
			relu_2 = tf.nn.relu(BN_2)
			dropout_2 = tf.layers.dropout(relu_2, drop, training=istraining)
			conv_2 = tf.layers.conv1d(inputs=dropout_2, 
									filters=channel,
									kernel_size=kernel_size, 
									strides=1, 
									padding='same', 
									activation=None,
									use_bias=False,
									kernel_initializer=self.initializer)
			Cin = inputs.get_shape().as_list()[-1]
			Cout = conv_2.get_shape().as_list()[-1]
			if Cin != Cout:
				inputs = tf.layers.conv1d(inputs=inputs, 
										filters=channel,
										kernel_size=1, 
										strides=1, 
										padding='same', 
										activation=None,
										use_bias=True,
										kernel_initializer=self.initializer)
			max_pooling_input = tf.layers.max_pooling1d(inputs=inputs, 
														pool_size=subsample_factor, 
														strides=subsample_factor, 
														padding='same')
			result = tf.add(max_pooling_input,conv_2)
			print(result)
			return result




