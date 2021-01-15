import tensorflow as tf
import AdamW 
import numpy as np
from utils import *
#################### Functions for Optimizer ####################

class Optimizer:

	def AdamOptimizer(self, loss, params):
		learning_rate, current_iteration, TRAIN_SCOPE, cosine_annealing = params
		train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=TRAIN_SCOPE)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			if cosine_annealing:
				lr = tf.train.cosine_decay_restarts(learning_rate,
													global_step=current_iteration,
													first_decay_steps=20,
													t_mul=2.0,	
													m_mul=1.0)
			else:
				lr = learning_rate
			optimizer = tf.train.AdamOptimizer(lr)
			gradients = optimizer.compute_gradients(loss, 
													var_list=train_vars, 
													colocate_gradients_with_ops=True)
			capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) \
								for grad, var in gradients if grad is not None]
			train_op = optimizer.apply_gradients(capped_gradients)
			return train_op


	def get_loss_sigmoidCE_multilabel(self, logits, labels, num_class):
		loss = tf.losses.sigmoid_cross_entropy(
					multi_class_labels=labels,
					logits=logits)
		return loss
	
	def get_loss_weightdecay(self, scope, isl2=True):
		train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
		losses = []
		for train_var in train_vars:
			if isl2:
				losses.append(tf.nn.l2_loss(train_var))
			else:
				losses.append(tf.reduce_sum(tf.abs(train_var)))

		loss = tf.reduce_sum(losses)
		return loss 

	def get_loss_weight_orthogonal(self, scope):
		train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
		losses = []
		for train_var in train_vars:
			if ('batch' not in train_var.name) & ('kernel' in train_var.name):
				W = train_var
				W = tf.reshape(W, [-1, W.get_shape()[-1]])
				I = tf.eye(W.get_shape().as_list()[-1])
				WT = tf.transpose(W)
				loss = tf.reduce_mean(tf.square(tf.matmul(WT, W) - I))
				losses.append(loss)
		loss = tf.reduce_sum(losses)
		return loss
	
	def get_loss_weight_freq_orthogonal(self, scope):
		train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
		losses = []
		for train_var in train_vars:
			if ('batch' not in train_var.name) & ('kernel' in train_var.name):
				W = self.fft(train_var)
				W = tf.reshape(W, [-1, W.get_shape()[-1]])
				I = tf.eye(W.get_shape().as_list()[-1], dtype=tf.complex64)
				WT = tf.transpose(W)
				loss = tf.reduce_mean(tf.square(tf.matmul(WT, W) - I))
				loss = tf.abs(loss)
				losses.append(loss)
		loss = tf.reduce_sum(losses)
		return loss
	
	def get_loss_weight_proj_orthogonal_freq(self, scope, mul_factor):
		train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
		z_real, z_imag = get_dft_mtx(16*mul_factor)
		losses = []
		
		T = 16
		proj_real = tf.get_variable('proj_real', shape=[T*mul_factor,T], dtype=tf.float32)
		proj_imag = tf.get_variable('proj_imag', shape=[T*mul_factor,T], dtype=tf.float32)
		proj = (proj_real, proj_imag)
		loss = tf.reduce_mean((proj_real-z_real)**2+(proj_imag-z_imag)**2)
		losses.append(loss)

		for i in range(len(train_vars)):
			train_var = train_vars[i]
			if ('batch' not in train_var.name) & ('kernel' in train_var.name) & (train_var.shape[0]==16):
				W_real, W_imag = self.projection_complex(train_var, proj, mul_factor=mul_factor, name='proj_layer') # T',C1,C2
				W_real = tf.reshape(W_real, [-1, W_real.get_shape().as_list()[-1]]) # T'C1,C2
				W_imag = tf.reshape(W_imag, [-1, W_imag.get_shape().as_list()[-1]])
				W_real_T = tf.transpose(W_real, [1,0]) # C2, T'C1
				W_imag_T = tf.transpose(W_imag, [1,0])
				
				W1 = (W_real_T, W_imag_T)
				W2 = (W_real, W_imag)
				W_real, W_imag = self.matmul_complex(W1, W2) # C2,C2
				I = tf.eye(W_real.get_shape().as_list()[0])
				loss = tf.reduce_mean(tf.square(tf.sqrt(W_real**2+W_imag**2)-I))
				losses.append(loss)

		loss = tf.reduce_sum(losses)
		return loss
	
	def get_loss_weight_proj_orthogonal(self, scope, mul_factor):
		train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
		losses = []
		T = 16
		proj_real = tf.get_variable('proj_real', shape=[T*mul_factor,T], dtype=tf.float32)
		proj_imag = tf.get_variable('proj_imag', shape=[T*mul_factor,T], dtype=tf.float32)
		proj = (proj_real, proj_imag)
		
		for i in range(len(train_vars)):
			train_var = train_vars[i]
			if ('batch' not in train_var.name) & ('kernel' in train_var.name) & (train_var.shape[0]==16):
				W_real, W_imag = self.projection_complex(train_var, proj, mul_factor=mul_factor, name='proj_layer') # T',C1,C2
				W_real = tf.reshape(W_real, [-1, W_real.get_shape().as_list()[-1]]) # T'C1,C2
				W_imag = tf.reshape(W_imag, [-1, W_imag.get_shape().as_list()[-1]])
				W_real_T = tf.transpose(W_real, [1,0]) # C2, T'C1
				W_imag_T = tf.transpose(W_imag, [1,0])
				
				W1 = (W_real_T, W_imag_T)
				W2 = (W_real, W_imag)
				W_real, W_imag = self.matmul_complex(W1, W2) # C2,C2
				I = tf.eye(W_real.get_shape().as_list()[0])
				
				loss = tf.reduce_mean(tf.square(tf.sqrt(W_real**2+W_imag**2)-I))
				losses.append(loss)
		loss = tf.reduce_sum(losses)
		return loss
	
	def fft_tmp(self, layer):
		B,T,C = layer.get_shape().as_list()
		layer = tf.transpose(layer, [0,2,1])
		layer = tf.reshape(layer, [B*C,T])
		feature = tf.fft(tf.cast(layer, tf.complex64))
		feature = tf.abs(feature)*2/T
		feature = tf.reshape(feature, [B,C,T])
		feature = feature[:,:,:int(T/2)]
		return feature
	
	def fft(self, weight):
		T,C1,C2 = weight.get_shape().as_list()
		weight = tf.reshape(weight, [T,C1*C2])
		T, _ = weight.get_shape().as_list()

		weight = tf.transpose(weight, [1,0])
		weight_f = tf.fft(tf.cast(weight, tf.complex64))
		weight_f = tf.transpose(weight_f, [1,0])
		weight_f = tf.reshape(weight_f, [T,C1,C2])
		return weight_f 

	def projection_complex(self, weight, proj, mul_factor=1, name='layer_proj'):
		with tf.variable_scope(name):
			T,C1,C2 = weight.get_shape().as_list()
			weight = tf.reshape(weight, [T,C1*C2])
			proj_real, proj_imag = proj

			w_real = tf.matmul(proj_real, weight)
			w_imag = tf.matmul(proj_imag, weight)
			w_real = tf.reshape(w_real, [T*mul_factor,C1,C2])
			w_imag = tf.reshape(w_imag, [T*mul_factor,C1,C2])
			
			return w_real, w_imag
	
	def matmul_complex(self, w1, w2):
		w1_real, w1_imag = w1
		w2_real, w2_imag = w2
		
		w_real1 = tf.matmul(w1_real, w2_real)
		w_imag1 = tf.matmul(w1_real, w2_imag)
		w_imag2 = tf.matmul(w1_imag, w2_real)
		w_real2 = -tf.matmul(w1_imag, w2_imag)
		
		w_real = w_real1 + w_real2
		w_imag = w_imag1 + w_imag2
		
		return w_real, w_imag
		

