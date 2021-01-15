import os
import tensorflow as tf
import numpy as np
import time
import csv
import math
import pickle as pkl
from utils import *
from random import shuffle
from tqdm import tqdm
import itertools
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA

flags = tf.app.flags.FLAGS


class Manager:
	def __init__(self, config, data, model):
		GPU = list(map(int, flags.GPU.split('/')))
		os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
		os.environ["CUDA_VISIBLE_DEVICES"]= ','.join(str(GPUi) for GPUi in GPU)

		self.epoch = 0
		self.graph = tf.get_default_graph()
		params_drop = list(map(float, flags.dropout.split('/')))
		self.drop = params_drop
		params_aug = list(map(float, flags.aug.split('/')))
		self.print_user_flags = config.print_user_flags

		if flags.istest:
			self.list_test = data.list_test
		else:
			self.list_train = data.list_train
			self.list_val = data.list_val

		self.input_generator = data.input_generator
		self.fetch_data = data.fetch_data
		
		self.X = model.X
		self.Y = model.Y
		self.dropout= model.dropout
		self.current_iteration = model.current_iteration
		self.istraining = model.istraining
		self.loss = model.loss
		self.train_op = model.train_op
		self.probs = model.probs
		self.loss_h = model.loss_h
		self.encoded = model.encoded
		self.session_launcher()

	##### Manager ##################################################
	def session_launcher(self):
		config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		config.gpu_options.allow_growth=True
		self.sess = tf.Session(config=config)
		self.variable_initializer(self.sess, flags.DIR_SOURCE)

	def variable_initializer(self, sess, DIR_SOURCE):
		vars_load = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
										scope=flags.LOAD_SCOPE)
		vars_load2 = []
		for i in range(len(vars_load)):
			if 'Adam' not in vars_load[i].name:
				vars_load2.append(vars_load[i])
		vars_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		saver_load = tf.train.Saver(vars_load2, max_to_keep=1)
		self.saver = tf.train.Saver(vars_save, max_to_keep=1)
		sess.run(tf.global_variables_initializer())
		try:
			ckpt = tf.train.get_checkpoint_state(DIR_SOURCE)
			print('ckpt:', ckpt)
			if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
				try:
					print('ckpt.model_checkpoint_path:',ckpt.model_checkpoint_path)
					saver_load.restore(sess, ckpt.model_checkpoint_path)
					print("Successfully loaded:",ckpt.model_checkpoint_path)
				except:
					print("Error on loading")
		except:
			print("Initialized w/o parameter loading")

	def weights_save(self, saver, DIR_SAVE, index):
		saver.save(sess=self.sess, save_path=DIR_SAVE+'model', global_step=index)
		print('Weights saved:', DIR_SAVE)
	
	def fit(self):
		self.history_writer()
		self.PBT = PopulationBasedTraining(self.sess)
		for self.epoch in range(flags.training_epochs):
			if flags.sample_search > 1:
				self.PBT.update()
			for j in range(flags.sample_search):
				#self.PBT.weights_load(j)
				for k in range(flags.search_iteration):
					print('Training', str(self.epoch), '-', j, '-', k)
					self.inference(params_iter = [j, k], 
									params_aug=self.PBT.augs_params,
									list_data=[self.list_train],
									iteration_fn=self.train_iter,
									istraining=True)
					self.inference(params_iter=[j, k], 
									params_aug=self.PBT.augs_params,
									list_data=[self.list_val],
									iteration_fn=self.test_iter,
									istraining=False)

	def inference_only(self):
		self.inference(params_iter=[-1, -1], 
						params_aug=[[-1,0,0],[-1,0,0]],
						list_data=[self.list_test],
						iteration_fn=self.test_iter,
						istraining=False)
	
	def inference(self, params_iter, params_aug, list_data, iteration_fn, istraining):
		# Data loader(generator)
		input_generators = []
		if istraining:
			list_input = list_data[0]
			shuffle(list_input)
		else:
			list_input = list_data[0]
		iter_num = int(len(list_input)/flags.batch_size)
		input_generators.append(self.input_generator(self.sess, list_input, flags.batch_size))
		
		j, k = params_iter
		aug = params_aug[j][0] + params_aug[j][1]
		self.result_initializer()
		for i in tqdm(range(iter_num)):
			# Input loading
			filename, data = self.fetch_data(self.sess, input_generators[0], istraining)
			# Inference 
			iteration_fn(self.sess, [filename, data], 
						params=[self.drop, aug, self.epoch], save=True)
		self.evaluation(self.sess, (self.epoch, j, k, params_aug), istraining)

	def train_iter(self, sess, data, params, save=True):
		pid, data = data
		dataX, dataY = data
		dataY = np.transpose(np.array(dataY), [1,0])
		drop, aug, epoch = params
		list_feed = {}
		list_feed[self.X] = dataX
		list_feed[self.Y] = dataY
		list_feed[self.dropout] = drop
		list_feed[self.current_iteration] = float(epoch)
		list_feed[self.istraining] = True
		_, prob, loss = sess.run([self.train_op, 
									self.probs,
									self.loss_h,
									self.encoded],
									feed_dict=list_feed)
		if save:
			self.result_collector(pid=pid, loss=loss, prob=prob, y=data)

	def test_iter(self, sess, data, params, save=True):
		pid, data = data
		dataX, dataY = data
		dataY = np.transpose(np.array(dataY), [1,0])
		drop, aug, epoch = params
		
		list_feed = {}
		list_feed[self.X] = dataX
		list_feed[self.Y] = dataY
		list_feed[self.dropout] = [0,0,0]
		list_feed[self.current_iteration] = float(epoch)
		list_feed[self.istraining] = False
		
		prob, loss = sess.run([self.probs,
								self.loss_h],
								feed_dict=list_feed)
		if save:
			self.result_collector(pid=pid, loss=loss, prob=prob, y=dataY)

	#### Result ##################################################
	def history_writer(self):
		if not os.path.exists(flags.DIR_SAVE):
			os.mkdir(flags.DIR_SAVE)
		self.history_saver('1. Configs', 'setting')
		self.history_saver(self.print_user_flags(), 'setting')
		self.history_saver('2. Model', 'setting')
		self.history_saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), 'setting')
	
	def history_saver(self, save_list, file_name, typ='a'):
		save_result = open(flags.DIR_SAVE+file_name+'.txt', typ)
		if isinstance(save_list, list):
			for save_index in save_list:
				save_result.write(str(save_index))
				save_result.write('\n')
		else:
			save_result.write(save_list)
			save_result.write('\n')
		save_result.close()
	
	def result_initializer(self):
		self.list_pid = []
		self.list_loss = []
		self.list_y = []
		self.list_prob = []

	def result_collector(self, pid, loss, y, prob, enc):
		self.list_pid.append(np.reshape(pid, [-1,1]))
		self.list_loss.append(np.reshape(loss, [-1,1]))
		self.list_y.append(np.reshape(y, [-1,flags.output_size]))
		self.list_prob.append(np.reshape(prob,[-1, flags.output_size]))

	def get_result(self, pid=False, loss=False, y=False, prob=False):
		if pid:		self.list_pid = np.concatenate(self.list_pid, axis=0)
		if loss:	self.list_loss = np.concatenate(self.list_loss, axis=0)
		if y:		self.list_y = np.concatenate(self.list_y, axis=0)
		if prob:	self.list_prob = np.concatenate(self.list_prob, axis=0)

	def evaluation(self, sess, params_iter, istraining):
		i, j, k, params_aug = params_iter
		current_training = str(i)+'-'+str(j)+'-'+str(k)
		
		if istraining:		step = 'train'
		elif flags.istest:	step = 'test'
		else:				step = 'val'
		
		self.get_result(pid=True, loss=True, y=True, prob=True)
		list2save = []
		if step == 'test':
			cutoff = list(np.load(flags.DIR_SAVE+'cutoff_val.npy'))
			(F,Fs,cutoff), (auroc,aurocs), (auprc,auprcs) \
				= classification_result_multilabel(self.list_y, self.list_prob, flags.output_size, cutoff)
		else:
			(F,Fs,cutoff), (auroc,aurocs), (auprc,auprcs) \
				= classification_result_multilabel(self.list_y, self.list_prob, flags.output_size)
			if step == 'train':	
				self.cutoff_tr = cutoff
			else:			
				self.cutoff_val = cutoff
		
		list2save.extend([current_training])
		loss = np.mean(self.list_loss)
		list2save.extend([loss])
		list2save.extend([F])
		list2save.extend([auroc])
		list2save.extend([auprc])
		self.history_saver(str(list2save)[1:-1], 'log-'+step, 'a')
		
		# Validation
		is step ==  'val':
			if i+j+k == 0: self.result = [0, -100] # Initialize results
			criterion = auprc
			if (self.result[-1]<criterion) & (i>=5) & (auroc!=0.5):
				self.result[0], self.result[1] = current_training, criterion
				self.weights_save(self.saver, flags.DIR_SAVE, i)
				np.save(flags.DIR_SAVE+'cutoff_val', self.cutoff_val)
				np.save(flags.DIR_SAVE+'cutoff_tr', self.cutoff_tr)
			print('### Best Model:', self.result[0], ':', self.result[-1])



class PopulationBasedTraining:
	def __init__(self, sess):
		self.sess = sess
		#self.result_cache = [0] * flags.sample_search
		#self.cache_load = [i for i in range(flags.sample_search)]
		#self.augs_params = [[[0,0,0],[0,0,0]] for j in range(flags.sample_search)]
		self.initializer()

	def weights_load(self, idx):
		DIR_SAVE = flags.DIR_SAVE+'cache/'
		ckpt = tf.train.get_checkpoint_state(DIR_SAVE)
		weight_name = DIR_SAVE+'model_cache-' + str(idx)
		print(weight_name)
		try:
			index = list(ckpt.all_model_checkpoint_paths).index(weight_name)
			self.saver.restore(self.sess, ckpt.all_model_checkpoint_paths[index])
			print('Weights loadd:', ckpt.all_model_checkpoint_paths[index])
		except:
			print('Weights name error')

	def weights_save(self, idx):
		DIR_SAVE = flags.DIR_SAVE+'cache/model_cache'
		self.saver.save(sess=self.sess, save_path=DIR_SAVE, global_step=idx)
		print('Weights saved:', DIR_SAVE)

	def exploit(self, result_cache):
		cache_rank = np.argsort(result_cache)
		cache_load = [i for i in range(flags.sample_search)]
		for j in range(int(flags.sample_search*0.25)):
			cache_load[cache_rank[j]] = cache_rank[-1-j]
		return cache_load

	def explore(self, augs_params):
		for j in range(flags.sample_search):
			aug_params = augs_params[j]
			for i in range(2):
				aug = aug_params[i]
				# 20%:random initialization, 80%:perturbation
				if np.random.random() < 0.2:
					aug[0] = np.random.randint(low=0, high=3+1)
					aug[1] = round(np.random.randint(low=0,high=10+1)/10,1)
					aug[2] = round(np.random.randint(low=1,high=10+1)/10,1)
				else:
					aug[1] = aug[1]+round(np.random.normal(loc=0, scale=0.1),1)
					aug[1] = np.clip(aug[1], 0, 1)
					aug[2] = aug[2]+round(np.random.normal(loc=0, scale=0.1),1)
					aug[2] = np.clip(aug[2], 1, 1)
				aug_params[i] = aug
			augs_params[j] = aug_params
		return augs_param

	def initializer(self):
		vars_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
		self.saver = tf.train.Saver(vars_save, max_to_keep=flags.sample_search+1)
		for i in range(flags.sample_search):
			self.weights_save(i)
		self.result_cache = [0] * flags.sample_search
		# Exploit initializatoin
		self.cache_load = [i for i in range(flags.sample_search)]
		# Explore initialization
		self.augs_params = [[[0,0,0],[0,0,0]] for j in range(flags.sample_search)]
		for j in range(flags.sample_search):
			for i in range(2):
				self.augs_params[j][i][0] = np.random.randint(low=0, high=3) # method
				self.augs_params[j][i][1] = np.round(np.random.randint(low=0,high=10+1)/10,1) # prob
				self.augs_params[j][i][2] = np.round(np.random.randint(low=1,high=10+1)/10,1) # intensity

	def update(self):
		self.cache_load = self.exploit(self.result_cache)
		self.augs_params = self.explore(self.augs_params)
	
