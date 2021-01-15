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
			params_init = list(map(int, flags.initializer.split('/')))
			self.initializer = tf.contrib.layers.variance_scaling_initializer(\
								factor=params_init[0], uniform=params_init[1])
			layer = tf.layers.conv1d(inputs=layer,
									filters=32,
									kernel_size=16,
									strides=2,
									padding='same',
									use_bias=True,
									kernel_initializer=self.initializer)
			channels = [16,24,24,40,40,80,80,80,112,112,112,192,192,192,192,320]
			upsample_factor = [1,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6]
			subsample_factor = [1,2,1,2,1,2,1,1,2,1,1,2,1,1,1,1,1,1]
			kernel_size = [9,9,9,25,25,9,9,9,25,25,25,25,25,25,25,9]
			for i in range(len(channels)):
				layer = self.inverted_bottleneck(\
								layer, kernel_size[i], channels[i], upsample_factor[i], subsample_factor[i], i)
			layer = tf.layers.conv1d(inputs=layer,
									filters=1280,
									kernel_size=1,
									strides=1,
									padding='same',
									use_bias=True,
									kernel_initializer=self.initializer)
			layer = tf.layers.batch_normalization(layer, training=self.istraining)
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
			print(logits)
			return logits
	
	def inverted_bottleneck(self, inputs, kernel_size, channel, upsample_factor, subsample_factor, i):
		istraining = self.istraining
		drop = self.dropout[1]
		with tf.variable_scope('inverted_block'+str(i)):
			conv_1 = tf.layers.conv1d(inputs=inputs,
									filters=inputs.get_shape().as_list()[-1]*upsample_factor,
									kernel_size=1,
									strides=1,
									padding='same',
									use_bias=False,
									kernel_initializer=self.initializer)
			BN_1 = tf.layers.batch_normalization(conv_1, training=istraining)
			relu_1 = tf.nn.relu6(BN_1)
			N,T,C = relu_1.get_shape().as_list()
			relu_1 = tf.reshape(relu_1, [N,T,1,C])
			conv_2 = tf.contrib.layers.separable_conv2d(
									inputs=relu_1,
									num_outputs=None,
									kernel_size=(kernel_size,1),
									depth_multiplier=1,
									stride=subsample_factor,
									padding='SAME',
									weights_initializer=self.initializer)
			N,T,_,C = conv_2.get_shape().as_list()
			conv_2 = tf.reshape(conv_2, [N,T,C])
			BN_2 = tf.layers.batch_normalization(conv_2, training=istraining)
			relu_2 = tf.nn.relu6(BN_2)
			conv_3 = tf.layers.conv1d(inputs=relu_2,
									filters=channel,
									kernel_size=1,
									strides=1,
									padding='same',
									use_bias=False,
									kernel_initializer=self.initializer)
			BN_3 = tf.layers.batch_normalization(conv_3, training=istraining)
			outputs = BN_3
			Cin = inputs.get_shape().as_list()[-1]
			Cout = BN_3.get_shape().as_list()[-1]
			if Cin == Cout:
				outputs = tf.add(inputs,outputs)
			return outputs
