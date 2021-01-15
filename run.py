import os
import sys
import tensorflow as tf
import numpy as np
import time
import csv
from config import *

cfg = Config()
import importlib 


import_fn = importlib.import_module
Data = import_fn(flags.data).Data
Build = import_fn(flags.build).Build
Manager = import_fn(flags.manager).Manager

if __name__=="__main__":

	if flags.istest:
		print('### TEST')
		data = Data(cfg)
		model = Build(cfg)
		manager = Manager(cfg, data, model)
		manager.inference_only()

	else:
		print('### TRAINING')
		data = Data(cfg)
		model = Build(cfg)
		manager = Manager(cfg, data, model)
		manager.fit()







