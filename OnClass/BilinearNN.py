import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
import sys
from scipy.special import softmax
import time
import math
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats

class BilinearNN:
	def __init__(self, Y_emb, nseen, ngene, use_pretrain = None, nhidden=[1000], l2=0.005):
		tf.reset_default_graph()
		self.ncls, self.ndim = np.shape(Y_emb)
		self.l2 = l2
		self.nseen = nseen
		self.ngene = ngene

		self.use_pretrain = use_pretrain
		self.seen_Y_emb = Y_emb[:nseen,:]
		self.nY = np.shape(self.seen_Y_emb)[0]
		self.seen_Y_emb = np.array(self.seen_Y_emb, dtype=np.float32)
		self.Y_emb = np.array(Y_emb, dtype=np.float32)

		self.nhidden = [self.ngene]
		self.nhidden.extend(nhidden)
		self.nhidden.append(self.ndim)
		tf.set_random_seed(3)

		self.__build()
		self.__build_loss()
		self.saver = tf.train.Saver()
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
		if self.use_pretrain is not None:
			self.saver.restore(self.sess, self.use_pretrain)
		#else:
		#	self.sess.run(tf.global_variables_initializer())


	def read_training_data(self, train_X, train_Y, use_valid = False, valid_X = None, valid_Y = None, test_X = None, test_Y = None):
		self.use_valid = use_valid
		if self.use_valid:
			if valid_X is None:
				np.random.seed(1)
				nx = np.shape(train_X)[0]
				ntrain = int(nx*0.9)
				permutation = list(np.random.permutation(nx))
				train_ind = permutation[:ntrain]
				valid_ind = permutation[ntrain:]
				self.train_X = train_X[train_ind, :]
				self.valid_X = train_X[valid_ind, :]
				self.train_Y = train_Y[train_ind]
				self.valid_Y = train_Y[valid_ind]
			else:
				self.train_X = train_X
				self.valid_X = valid_X
				self.train_Y = train_Y
				self.valid_Y = valid_Y
			self.valid_Y = self.one_hot_matrix(self.valid_Y, self.ncls)
		else:
			self.train_X = train_X
			self.train_Y = train_Y
		self.train_Y = self.one_hot_matrix(self.train_Y, self.nseen)
		self.nX = np.shape(self.train_X)[0]

		if test_X is not None:
			self.test_X = test_X
			self.test_Y = self.one_hot_matrix(test_Y, self.ncls)
		else:
			self.test_X = None
			self.test_Y = None
		self.train_X = np.array(self.train_X, dtype=np.float32)
		self.train_Y = np.array(self.train_Y, dtype=np.float32)


	def __build(self,stddev=0.0001, seed=3):#W,H,B
		tf.set_random_seed(seed) # set seed to make the results consistant
		#w_init =  tf.contrib.layers.xavier_initializer(seed = seed)
		w_init =  tf.glorot_uniform_initializer(seed = seed)#tf 2.0 has no tf.contribu. Glorot is the same as xavier
		b_init = tf.zeros_initializer()
		self.nlayer = len(self.nhidden)
		self.W = {}
		self.B = {}
		for i in range(1,self.nlayer):
			self.W[i] = tf.get_variable("W"+str(i), [self.nhidden[i], self.nhidden[i-1]], initializer = w_init, dtype=tf.float32)
			self.B[i] = tf.get_variable("B"+str(i), [self.nhidden[i], 1], initializer = b_init, dtype=tf.float32)
		#self.B = tf.get_variable("B", [1, 1], initializer = tf.zeros_initializer(), dtype=tf.float32)

	def __build_loss(self):
		self.mini_train_X = tf.placeholder(shape=[None, self.ngene], dtype=tf.float32)
		self.mini_train_Y = tf.placeholder(shape=[None, self.nseen], dtype=tf.float32)
		self.mini_train_Y_emb = tf.placeholder(shape=[None, None], dtype=tf.float32)
		self.train_keep_prob = tf.placeholder(tf.float32)
		self.p = self.mini_train_X
		self.loss = 0
		for i in range(1,self.nlayer):
			self.p = tf.add(tf.matmul(self.p, tf.transpose(self.W[i])), tf.transpose(self.B[i]))
			if i != self.nlayer - 1:
				self.p = tf.nn.relu(self.p)
				self.p = tf.nn.dropout(self.p, self.train_keep_prob)
			self.loss += (tf.nn.l2_loss(self.W[i])) * self.l2
		self.p = tf.matmul(self.p, tf.transpose(self.mini_train_Y_emb))
		self.label = self.mini_train_Y
		self.entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.p, labels = self.label)
		self.loss += tf.reduce_mean(self.entropy)

	# Turn labels into matrix
	def one_hot_matrix(self, labels, C):
		C = tf.constant(C, name = "C")
		one_hot_matrix = tf.one_hot(labels, C, axis = 0)
		sess_tmp = tf.Session()
		one_hot = sess_tmp.run(one_hot_matrix)
		sess_tmp.close()
		return one_hot.T

	def predict(self, test_X):
		test_X = np.array(test_X, dtype=np.float32)
		p = self.sess.run(self.p, feed_dict={self.mini_train_X:test_X, self.mini_train_Y_emb: self.seen_Y_emb, self.train_keep_prob: 1.0})
		test_Y_pred = softmax(p, axis=1)
		return test_Y_pred

	def predict_prob(self,  X, Y, Y_emb):
		label = Y
		p = self.sess.run(self.p, feed_dict={self.mini_train_X: X, self.mini_train_Y_emb: Y_emb, self.train_keep_prob: 1.0})
		p = softmax(p, axis=1)
		accuracy = np.mean(np.argmax(p,axis=1) == np.argmax(label,axis=1))
		[nsample, nclass] = np.shape(Y)
		class_auc_macro = np.full(nclass, np.nan)
		class_auprc_macro =  np.full(nclass, np.nan)
		for i in range(self.nseen):
			if len(np.unique(Y[:,i]))==2:
				class_auc_macro[i] = roc_auc_score(Y[:,i], p[:,i])
				#auc_tmp = roc_auc_score(Y[:,i], p[:,i]*1e50)
				class_auprc_macro[i] = average_precision_score(Y[:,i], p[:,i])
				#auprc_tmp = average_precision_score(Y[:,i], p[:,i]*1e50)
				#print (class_auprc_macro[i],  class_auc_macro[i])
		seen_auroc = np.nanmedian(class_auc_macro)
		seen_auprc = np.nanmedian(class_auprc_macro)
		class_auc_macro = np.full(nclass, np.nan)
		class_auprc_macro =  np.full(nclass, np.nan)
		for i in range(self.nseen, self.ncls):
			if i >= np.shape(Y)[1] or i  >= np.shape(p)[1]:
				break
			if len(np.unique(Y[:,i]))==2:
				class_auc_macro[i] = roc_auc_score(Y[:,i], p[:,i])
				class_auprc_macro[i] = average_precision_score(Y[:,i], p[:,i])
		unseen_auroc = np.nanmedian(class_auc_macro)
		unseen_auprc = np.nanmedian(class_auprc_macro)

		return accuracy, p, seen_auroc, seen_auprc, unseen_auroc, unseen_auprc

	# Get the mini batches
	def random_mini_batches(self, X, Y, mini_batch_size=32, seed=1):
		# input -- X (training set), Y (true labels)
		# output -- mini batches
		ns = X.shape[0]
		mini_batches = []
		np.random.seed(seed)
		# shuffle (X, Y)
		permutation = list(np.random.permutation(ns))
		shuffled_X = X[permutation, :]
		shuffled_Y = Y[permutation, :]
		# partition (shuffled_X, shuffled_Y), minus the end case.
		num_complete_minibatches = int(math.floor(ns/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
		for k in range(0, num_complete_minibatches):
			mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
			mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)
		# handling the end case (last mini-batch < mini_batch_size)
		if ns % mini_batch_size != 0:
			mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : ns,:]
			mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : ns,:]
			mini_batch = (mini_batch_X, mini_batch_Y)
			mini_batches.append(mini_batch)
		return mini_batches

	def optimize(self, max_iter = 50, minibatch_size = 128, keep_prob = 1., lr = 0.0001, save_model = None):

		self.keep_prob = keep_prob
		tf.set_random_seed(3)
		seed = 3
		global_step = tf.Variable(0, trainable=False)
		decay_lr = tf.train.exponential_decay(lr, global_step, 1000, 0.95, staircase=True)
		train_op =  tf.train.AdamOptimizer(learning_rate=decay_lr).minimize(self.loss)
		#print (self.sess.run(self.W[1]))
		self.sess.run(tf.global_variables_initializer())
		for epoch in range(max_iter):
			seed = seed + 1
			minibatches = self.random_mini_batches(self.train_X, self.train_Y, minibatch_size, seed)
			epoch_cost = 0.
			num_minibatches = int(self.nX / minibatch_size)
			for minibatch in minibatches:
				(minibatch_X, minibatch_Y) = minibatch
				minibatch_cost, p, label, entropy,  _ = self.sess.run([self.loss, self.p, self.label, self.entropy, train_op], feed_dict={self.train_keep_prob: self.keep_prob,self.mini_train_X: minibatch_X, self.mini_train_Y: minibatch_Y, self.mini_train_Y_emb: self.seen_Y_emb})
				epoch_cost += minibatch_cost/ num_minibatches

			if (epoch+1) % 1 == 0:
				train_acc,_,train_auroc, train_auproc,_,_ = self.predict_prob(self.train_X, self.train_Y, self.seen_Y_emb)
				print ("Training cost after epoch %i: loss:%.6f acc: %.3f auc: %.3f auprc: %.3f" % (epoch+1, epoch_cost,train_acc,train_auroc,train_auproc))
				sys.stdout.flush()
				if self.use_valid:
					valid_acc,_,valid_auroc,valid_auprc, valid_unseen_auroc, valid_unseen_auprc = self.predict_prob(self.valid_X, self.valid_Y, self.Y_emb)
					print ("Validation cost after epoch %i: acc: %.3f auc: %.3f auprc: %.3f unseen_auc : %.3f unseen_auprc: %.3f" % (epoch+1, valid_acc,valid_auroc,valid_auprc, valid_unseen_auroc, valid_unseen_auprc))
				if self.test_X is not None:
					test_acc,_,test_auroc,test_auprc, test_unseen_auroc, test_unseen_auprc = self.predict_prob(self.test_X, self.test_Y, self.Y_emb)
					print ("Test cost after epoch %i: acc: %.3f auc: %.3f auprc: %.3f unseen_auc : %.3f unseen_auprc: %.3f" % (epoch+1, test_acc,test_auroc,test_auprc,test_unseen_auroc,test_unseen_auprc))
		if save_model is not None:
			self.saver.save(self.sess, save_model)
