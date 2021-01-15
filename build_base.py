import os
import sys
import time
import csv
import math
import random
import numpy as np
import tensorflow as tf
from preprocessing2 import *
from utils import *
flags = tf.app.flags.FLAGS

# Import Model
import importlib
import_fn = importlib.import_module
Model = import_fn(flags.model).Model
# Import Optimizer
from optimizer import Optimizer

class Build(Model, Optimizer):
	def __init__(self, config):
		#self.model = Model.build_model
		self.build_graph()
	
	def build_graph(self):
		self.X = tf.placeholder(tf.float32, [None, int(flags.frequency*flags.length)*12]) 
		self.Y = tf.placeholder(tf.int32, [None, flags.output_size])
		self.Z = tf.placeholder(tf.int32, [None, 2])
		self.dropout = tf.placeholder(tf.float32, 3)
		self.current_iteration = tf.placeholder(tf.float32)
		self.istraining = tf.placeholder(tf.bool)

		X = tf.split(self.X, flags.GPU_NUM)
		Y = tf.split(self.Y, flags.GPU_NUM)
		Z = tf.split(self.Z, flags.GPU_NUM)
		
		probs = []
		losses_class = []
		losses_weightdecay = []
		losses_contrastive = []
		losses_features = []
		losses_features_b = []
		losses_ae = []
		losses_cluster = []

		for i in range(flags.GPU_NUM):
			with tf.device('gpu:%d' %i):
				x, y, z = X[i], Y[i], Z[i]
				#params = (flags.aug_base, flags.aug_graph, self.istraining, flags.batch_size_gpu)
				x_base = preprocessing(x, self.istraining) # N, T, C

				# BASE FORWARD
				with tf.variable_scope(flags.name, reuse=tf.AUTO_REUSE):
					params = (self.dropout, self.istraining, flags.batch_size_gpu)
					encoded = self.build_encoder(x_base, params, 'encoder')
					self.encoded = encoded
					# RESULT
					outputs = self.build_classifier(encoded, params, 'classifier')
					probs.append(tf.nn.sigmoid(outputs))
			
					lcls = self.get_loss_sigmoidCE_multilabel(outputs, y, flags.output_size)
					if flags.orthogonal == 'base':
						lw_e = self.get_loss_weight_orthogonal(flags.TRAIN_SCOPE+'/encoder')
					elif flags.orthogonal == 'freq':
						lw_e = self.get_loss_weight_freq_orthogonal(flags.TRAIN_SCOPE+'/encoder')
					elif flags.orthogonal == 'proj_freq':
						lw_e = self.get_loss_weight_proj_orthogonal_freq(flags.TRAIN_SCOPE+'/encoder', flags.proj_scale)
					elif flags.orthogonal == 'proj':
						lw_e = self.get_loss_weight_proj_orthogonal(flags.TRAIN_SCOPE+'/encoder', flags.proj_scale)
					else:
						lw_e = self.get_loss_weightdecay(flags.TRAIN_SCOPE+'/endocer', False)
					lw_cls = self.get_loss_weightdecay(flags.TRAIN_SCOPE+'/classifier', False)
				
				losses_class.append(lcls)
				losses_weightdecay.append(lw_e*flags.reg_weightdecay)
				losses_weightdecay.append(lw_cls*1e-5)

		# LOSS 
		self.loss_class = tf.reduce_mean(losses_class)
		self.loss_weightdecay = tf.reduce_sum(losses_weightdecay)

		# Pretrain(Unsupervised)
		self.loss = self.loss_weightdecay
		self.loss += self.loss_class
		self.loss_h = self.loss - self.loss_weightdecay*flags.reg_weightdecay
		
		# OPTIMIZER 
		self.probs = tf.concat([probs[i] for i in range(flags.GPU_NUM)], axis=0)
		params = (flags.learning_rate, self.current_iteration, flags.TRAIN_SCOPE, False)
		if flags.optimizer == 'adam':
			self.train_op = self.AdamOptimizer(self.loss, params)
		elif flags.optimizer == 'adam2':
			self.train_op = self.AdamOptimizer_v2(self.loss, params, flags.model)
		elif flags.optimizer == 'sgd':
			self.train_op = self.SGDOptimizer(self.loss, params)
		elif flags.optimizer == 'adamw':
			self.loss = self.loss - self.loss_weightdecay
			self.train_op = self.AdamWOptimizer(self.loss, params, flags.reg_weightdecay)
			#self.train_op = tf.contrib.opt.AdamWOptimizer(weight_decay = 0.01,learning_rate=self.lr).minimize(self.loss_c)









#	#################### Functions for Optimizer ####################
#	
#	def AdamOptimizer(self, cosine_annealing=True):
#		train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=flags.TRAIN_SCOPE)
#		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#		with tf.control_dependencies(update_ops):
#			if cosine_annealing:
#				lr = tf.train.cosine_decay_restarts(flags.learning_rate,
#																						global_step=self.current_iteration,
#																						first_decay_steps=20,
#																						t_mul=2.0,	
#																						m_mul=1.0)
#			else:
#				lr = flags.learning_rate
#			optimizer = tf.train.AdamOptimizer(lr)
#			gradients = optimizer.compute_gradients(self.loss, 
#																							var_list=train_vars, 
#																							colocate_gradients_with_ops=True)
#			capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) \
#														for grad, var in gradients if grad is not None]
#			train_op = optimizer.apply_gradients(capped_gradients)
#			return train_op
#
#	def GDOptimizer(self, cosine_annealing=True):
#		train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=flags.TRAIN_SCOPE)
#		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#		with tf.control_dependencies(update_ops):
#			if cosine_annealing:
#				lr = tf.train.cosine_decay_restarts(flags.learning_rate,
#																						global_step=self.current_iteration,
#																						first_decay_steps=20,
#																						t_mul=2.0,
#																						m_mul=1.0)
#			else:
#				lr = flags.learning_rate
#			optimizer = tf.train.GradientDescentOptimizer(lr)
#			gradients = optimizer.compute_gradients(self.loss, 
#																							var_list=train_vars, 
#																							colocate_gradients_with_ops=True)
#			capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) \
#														for grad, var in gradients if grad is not None]
#			train_op = optimizer.apply_gradients(capped_gradients)
#			return train_op
#	
#
#
#
#
#
#	#################### Functions for Loss ####################
#	def get_loss_class(self, logits, labels, num_class=flags.output_size, mask=None):
#		loss = tf.losses.softmax_cross_entropy(
#												onehot_labels=tf.one_hot(labels,num_class),
#												logits=logits,
#												label_smoothing=0.0)
#		if mask is not None:
#			loss = loss*mask
#			loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
#		else:
#			loss = tf.reduce_mean(loss)
#		return loss
#	
#	def get_loss_contrastive(self, x, y, size, T=1):
#		cor_mtx = tf.matmul(x, tf.transpose(y))
#		cor_mtx_diag = tf.eye(size)
#		loss_pos = tf.reduce_sum(-cor_mtx/T*cor_mtx_diag)/size
#		loss_neg = tf.reduce_sum(tf.log(tf.reduce_sum(tf.exp(cor_mtx/T)*(1-cor_mtx_diag),1)))/size
#		return loss_pos + loss_neg
#
#	def get_loss_weightdecay(self, scope):
#		train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
#		losses = []
#		for train_var in train_vars:
#			#losses.append(self.distance_riemann(train_var, tf.zeros_like(train_var)))
#			losses.append(tf.nn.l2_loss(train_var))
#		loss = tf.reduce_sum(losses)
#		#losses_reg.append(loss)
#		return loss 
#
#	def get_loss_ae(self, X_org, X_recon):
#		# X: N, Channel, Time
#		loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum((X_org - X_recon)**2, axis=-1)))
#		#losses_ae.append(loss)
#		return loss 
#
#	def get_loss_KL(self, rho, rho_cap):
#		# KL(rho||rho_cap)
#		loss = tf.reduce_sum(rho*tf.log(rho/rho_cap) 
#											+(1-rho)*tf.log((1-rho)/(1-rho_cap)))	
#		#losses_KL.append(loss)
#		return loss 
#	
#	def get_loss_orthogonal(self, losses_ortho):
#		# TODO: SCOPE of train_vars
#		train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=flags.TRAIN_SCOPE)
#		losses = []
#		for train_var in train_vars:
#			if 'orthogonal' in train_var.name:
#				W = train_var
#				W = tf.reshape(W, [-1, W.get_shape()[-1]])
#				WT = tf.transpose(W)
#				loss = tf.reduce_mean(tf.square(tf.matmul(WT, W) - I))
#				losses.append(loss)
#		losses_ortho.append(tf.reduce_sum(losses))
#		return losses_ortho
#
#	def GradNorm(self, loss_list, shared_layer):
#		# GradNorm:Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks 
#		# TODO
#		loss_base = tf.constant([0.7, 20])
#		losses = loss_list/loss_base
#		r = losses/tf.reduce_mean(losses)
#		alpha = 0.16
#		grads = []
#		for i in range(len(loss_list)):
#			grads.append(tf.reduce_mean(tf.square(tf.gradients(loss_list[i], shared_layer))))
#		grads_mean = tf.reduce_mean(grads, 0)
#		ratios = tf.abs(grads - grads_mean*(r**alpha))/loss_list
#		ratios = tf.stop_gradient(ratios)
#
#		#ratios = grads/grads_mean - tf.reduce_mean(grads/grads_mean)*r**alpha
#		#ratios = tf.stop_gradient(ratios)
#		#L = tf.gradients(L, shared_layer)
#		#tf.stop_gradient(L)
#		
#		#grad_mean = tf.reduce_mean(grads)
#		#alpha = 1
#		#L = tf.reduce_sum(tf.abs(grads - grad_mean * r**alpha))
#		#gradientL = tf.gradients(L, shared_layer)
#		
#		#ratios = grads/tf.reduce_sum(grads)
#		#ratios = tf.clip_by_value(1/ratios, 0, 10)
#		#ratios = tf.stop_gradient(ratios)
#		return ratios
#
#
#
#
#
#
#
