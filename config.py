import tensorflow as tf
#from utils import *

flags = tf.app.flags.FLAGS


### GPU, label, label_onehot, name, dir_save
class Config:
	def __init__(self):
		self.user_flags = []
		self.get_config()
	
	def get_config(self):
		DEFINE_string = self.DEFINE_string
		DEFINE_integer = self.DEFINE_integer
		DEFINE_float = self.DEFINE_float
		DEFINE_boolean = self.DEFINE_boolean
		
		DEFINE_string('manager', 'manager', '')
		DEFINE_string('data', 'data', '')
		DEFINE_string('build', 'build_base', '')
		DEFINE_string('model', 'model_resnet', '')
		DEFINE_string('DIR_SAVE', './save/', '')

		## GPU
		DEFINE_string('GPU', '6', '')
		DEFINE_integer('GPU_NUM', len(flags.GPU.split(',')),'')
		
		## Loading/Saving
		DEFINE_boolean('istest', True, '')
		DEFINE_string('name', 'NN', 'name of model')
		DEFINE_string('path_label', './data/index_with_tfrecords_icbeb_featextract31.pkl', '')
		DEFINE_string('path_data', './data/tfrecords_icbeb/', '')
		DEFINE_string('TRAIN_SCOPE', flags.name, '')
		
		if flags.istest:
			DEFINE_string('DIR_SOURCE', flags.DIR_SAVE, '')
			DEFINE_string('LOAD_SCOPE', flags.name, '')
		else:
			DEFINE_string('DIR_SOURCE', 'None', '')
			DEFINE_string('LOAD_SCOPE', flags.name, '')
		DEFINE_string('FILE_HISTORY', 'history.csv', '')
		
		## Data setting
		DEFINE_string('label', 'normal/af/avb/lbbb/rbbb/pac/pvc/std/ste', '')
		DEFINE_integer('num_lead', 12, 'the number of leads')

		## Pre-preocessing
		DEFINE_float('frequency', 250, 'Hz')
		DEFINE_float('length', 10, '')
		
		## Training setting 
		DEFINE_string('optimizer', 'adam', 'adam/sgd')
		DEFINE_float('learning_rate', 1e-3, '')
		DEFINE_integer('training_epochs', 150, '')
		
		#else:
		DEFINE_integer('batch_size', 32, '')
		DEFINE_integer('batch_size_gpu', int(flags.batch_size/flags.GPU_NUM), '')
		DEFINE_integer('sample_search', '1', '')
		DEFINE_integer('search_iteration', '1', '')
		DEFINE_integer('labeled_number', -1,'')
		DEFINE_string('initializer', '1/1', 'scale factor/uniform')

		## Regularization setting
		DEFINE_float('reg_weightdecay', 1e-5, 'weight decay(l2 loss)')
		DEFINE_string('orthogonal', 'freq', '')
		DEFINE_integer('proj_scale', 1, '')
		DEFINE_string('dropout', '0.15/0.15/0.5', 'encoder1/encoder2/fc')
		DEFINE_string('aug', '2/0.5' ,'number/mag')
		DEFINE_boolean('aug_base', True, '')
		DEFINE_integer('output_size', 9, '')



	def DEFINE_string(self, name, default_value, doc_string):
		tf.app.flags.DEFINE_string(name, default_value, doc_string)
		self.user_flags.append(name)

	def DEFINE_integer(self, name, default_value, doc_string):
		tf.app.flags.DEFINE_integer(name, default_value, doc_string)
		self.user_flags.append(name)

	def DEFINE_float(self, name, defualt_value, doc_string):
		tf.app.flags.DEFINE_float(name, defualt_value, doc_string)
		self.user_flags.append(name)

	def DEFINE_boolean(self, name, default_value, doc_string):
		tf.app.flags.DEFINE_boolean(name, default_value, doc_string)
		self.user_flags.append(name)

	def print_user_flags(self, line_limit = 80):
		temp = []
		for flag_name in sorted(self.user_flags):
			value = "{}".format(getattr(flags, flag_name))
			log_string = flag_name
			log_string += "." * (line_limit - len(flag_name) - len(value))
			log_string += value
			temp.append(log_string)
			print(log_string)
		return temp
