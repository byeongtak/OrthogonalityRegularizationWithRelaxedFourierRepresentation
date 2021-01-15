import sys
import os
import tensorflow as tf
import numpy as np
import time
import csv
import math
import pickle as pkl
import random
import math
import utils
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc
from sklearn.manifold import TSNE
from scipy.stats import moment
from scipy.stats import linregress
#from model import *
#from utils import *
#from utils_input import *
#from tensorflow.python.framework import ops
#from tensorflow.python.ops import nn_ops, gen_nn_ops

encoding = 'utf-8'
flags = tf.app.flags.FLAGS

class Data:
	def __init__(self, config):
		# ICBEB
		list_collect = self.split_dataset('label')		
		self.list_train, self.list_val, self.list_test = list_collect

	def split_dataset(self, criterion):
		df = pkl.load(open(flags.path_label, 'rb')) # index file
		self.df_dict = df.set_index('filename').to_dict('index')
		df.filename = flags.path_data + df.filename
		
		# pid separation
		pids_collect = self.separate_pids(df)
		# index extraction
		indicies_collect = self.extract_indicies(df, pids_collect)
		# dataset listing
		list_collect = self.get_list_of_label(df, indicies_collect)
		return list_collect

	def separate_pids(self, df):
		pids = sorted(list(set(df.PID)))
		random.seed(719)
		random.shuffle(pids)
		pids_train = pids[:int(len(pids)*0.7)]
		pids_val = pids[int(len(pids)*0.7):int(len(pids)*0.85)]
		pids_test = list(set(pids)-set(pids_train)-set(pids_val))
		pids_train = pids_train[:flags.labeled_number]
		return pids_train, pids_val, pids_test

	def extract_indicies(self, df, pids):
		pids_train, pids_val, pids_test, pids_unsupervised = pids
		idx_train = df[df.PID.isin(pids_train)].index
		idx_val = df[df.PID.isin(pids_val)].index
		idx_test = df[df.PID.isin(pids_test)].index
		return idx_train, idx_val, idx_test

	def get_list_of_label(self, df, idx):
		idx_train, idx_val, idx_test, idx_unsupervised = idx
		list_train = list(df.loc[idx_train].filename)
		list_val = list(df.loc[idx_val].filename)
		list_test = list(df.loc[idx_test].filename)
		return list_train, list_val, list_test

	## Input pipeline
	def parse_100(self, serialized):
		features = \
		{
			'waveform': tf.FixedLenFeature([5000*12,], tf.float32),
			'filename': tf.FixedLenFeature([], tf.string)
		}
		parsed = tf.parse_single_example(serialized=serialized, features=features)
		return parsed['waveform'], parsed['filename']
	
	def input_fn(self, filenames, batch_size, parse):
		dataset = tf.data.TFRecordDataset(filenames=filenames)
		dataset = dataset.map(parse)
		dataset = dataset.batch(batch_size)
		iterator = dataset.make_initializable_iterator()
		return iterator

	def input_generator(self, sess, datalist, batch_size):
		generator = self.input_fn(datalist, batch_size, self.parse_100)
		batch = generator.get_next()
		sess.run(generator.initializer)
		return batch

	def get_label(self, filenames, criterion):
		label = [] 
		for i in range(len(filenames)):
			filename = filenames[i].decode(encoding)# + '.tfrecords'
			if not filename.endswith('tfrecords'):
				filename = filename + '.tfrecords'
			label.append(self.df_dict[filename][criterion]*1)
		return np.array(label)
	
	## Fetching data
	def fetch_data(self, sess, input_generator, istraining=True):
		waveform, filename = sess.run(input_generator)
		waveform = self.preprocessing(waveform, istraining)
		labels = flags.label.split('/')
		label = []
		for i in range(len(labels)):
			label.append(self.get_label(filename, labels[i]))
		return filename, (waveform, label)
	
	def preprocessing(self, inputs, istraining=True):
		#data_length, sampling_length, sampling_num = params
		data_length = 10
		sampling_length = flags.length
		sampling_num = 1
		outputs = []
		for j in range(sampling_num):
			for i in range(len(inputs)):
				signal = inputs[i]
				signal = np.reshape(signal, [12, int(500*data_length)]) 
				waveform = signal[:,::2]
				waveform = np.reshape(waveform, [-1])
				outputs.append(waveform)
		return outputs

