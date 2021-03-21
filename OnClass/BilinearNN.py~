import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
from scipy.special import softmax
import time
import math
from sklearn.metrics import roc_auc_score
from scipy import stats
class BilinearNN:
	def __init__(self):
		#
		return

	def train(self, train_X, train_Y, Y_emb, nseen,  nhidden=[100], max_iter=500, save_model = None, minibatch_size=32, lr = 0.0001, l2=0.01, use_pretrain= None):
		tf.reset_default_graph()
		self.save_model = save_model
		self.use_pretrain = use_pretrain
		self.train_X = train_X
		self.train_Y = self.one_hot_matrix(train_Y, nseen)
		self.minibatch_size = minibatch_size
		self.nseen = nseen

		train_Y_emb = Y_emb[:nseen,:]
		self.nX,self.dimX = np.shape(train_X)
		self.nY,self.dimY = np.shape(train_Y_emb)
		#train_Y_emb = np.random.rand(self.nY,2000)
		self.train_Y_emb = train_Y_emb
		self.nY,self.dimY = np.shape(train_Y_emb)
		self.lr = lr
		self.l2 = l2
		self.max_iter = max_iter
		self.nhidden = [self.dimX]
		self.nhidden.extend(nhidden)
		self.nhidden.append(self.dimY)
		self.train_X = np.array(self.train_X, dtype=np.float64)
		self.train_Y = np.array(self.train_Y, dtype=np.float64)
		self.train_Y_emb = np.array(self.train_Y_emb, dtype=np.float64)
		self.Y_emb = np.array(Y_emb, dtype=np.float64)
		self.__build()
		self.__build_loss()
		self.optimize()
		train_Y_pred = self.predict_score(self.train_X, self.train_Y_emb)
		return train_Y_pred

	def predict(self, test_X, reset_model=True):
		test_X = np.array(test_X, dtype=np.float64)
		test_Y_pred = self.predict_score(test_X, self.Y_emb)
		if reset_model:
			tf.reset_default_graph()
		return test_Y_pred



	def __build(self,stddev=0.0001, seed=1):#W,H,B
		tf.set_random_seed(seed) # set seed to make the results consistant
		w_init =  tf.contrib.layers.xavier_initializer(seed = seed)
		b_init = tf.zeros_initializer()
		self.nlayer = len(self.nhidden)
		self.W = {}
		self.B = {}
		for i in range(1,self.nlayer):
			self.W[i] = tf.compat.v1.get_variable("W"+str(i), [self.nhidden[i-1], self.nhidden[i]], initializer = w_init, dtype=tf.float64)
			self.B[i] = tf.compat.v1.get_variable("B"+str(i), [1, self.nhidden[i]], initializer = b_init, dtype=tf.float64)
		#self.B = tf.get_variable("B", [1, 1], initializer = tf.zeros_initializer(), dtype=tf.float64)

	def __build_loss(self):
		self.mini_train_X = tf.compat.v1.placeholder(shape=[None, self.dimX], dtype=tf.float64)
		self.mini_train_Y = tf.compat.v1.placeholder(shape=[None, self.nseen], dtype=tf.float64)
		self.mini_train_Y_emb = tf.compat.v1.placeholder(shape=[None, None], dtype=tf.float64)
		self.p = self.mini_train_X
		self.loss = 0
		for i in range(1,self.nlayer):
			self.p = tf.add(tf.matmul(self.p, self.W[i]), self.B[i])
			if i != self.nlayer - 1:
				self.p = tf.nn.relu(self.p)
			self.loss += (tf.nn.l2_loss(self.W[i])) * self.l2
		self.p = tf.matmul(self.p, tf.transpose(self.mini_train_Y_emb))
		self.label = self.mini_train_Y
		self.entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.p, labels = self.mini_train_Y)
		self.loss += tf.reduce_mean(self.entropy)

	# Turn labels into matrix
	def one_hot_matrix(self, labels, C):
		C = tf.constant(C, name = "C")
		one_hot_matrix = tf.one_hot(labels, C, axis = 0)
		sess_tmp = tf.Session()
		one_hot = sess_tmp.run(one_hot_matrix)
		sess_tmp.close()
		return one_hot.T

	def predict_score(self, X, Y_emb):
		p = self.sess.run(self.p, feed_dict={self.mini_train_X: X, self.mini_train_Y_emb: Y_emb})
		score = softmax(p, axis=1)
		return score

	def predict_prob(self,  X, Y, Y_emb):
		label = Y
		p = self.sess.run(self.p, feed_dict={self.mini_train_X: X, self.mini_train_Y_emb: Y_emb})
		p = softmax(p, axis=1)
		accuracy = np.mean(np.argmax(p,axis=1) == np.argmax(label,axis=1))
		[nsample, nclass] = np.shape(Y)
		class_auc_macro = np.full(nclass, np.nan)
		for i in range(nclass):
			if len(np.unique(Y[:,i]))==2:
				class_auc_macro[i] = roc_auc_score(Y[:,i], p[:,i])
		auroc = np.nanmean(class_auc_macro)
		return accuracy, p, auroc

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

	def optimize(self):
		cost_val = []

		global_step = tf.Variable(0, trainable=False)
		decay_lr = tf.compat.v1.train.exponential_decay(self.lr, global_step, 1000, 0.95, staircase=True)

		train_op =  tf.compat.v1.train.AdamOptimizer(learning_rate=decay_lr).minimize(self.loss)
		if self.save_model is not None and self.use_pretrain is None:
			saver = tf.train.Saver()
		if self.use_pretrain is not None:
			saver = tf.train.Saver()
		self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True)))
		if self.use_pretrain is not None:
			saver.restore(self.sess, self.use_pretrain)
		else:
			self.sess.run(tf.compat.v1.global_variables_initializer())
			for epoch in range(self.max_iter):
				minibatches = self.random_mini_batches(self.train_X, self.train_Y, self.minibatch_size, epoch)
				epoch_cost = 0.
				num_minibatches = int(self.nX / self.minibatch_size)
				for minibatch in minibatches:
					(minibatch_X, minibatch_Y) = minibatch
					minibatch_cost, p, label, entropy,  _ = self.sess.run([self.loss, self.p, self.label, self.entropy, train_op], feed_dict={self.mini_train_X: minibatch_X, self.mini_train_Y: minibatch_Y, self.mini_train_Y_emb: self.train_Y_emb})
					epoch_cost += minibatch_cost
				if (epoch+1) % 1 == 0:
					train_acc,_,train_auroc = self.predict_prob(self.train_X, self.train_Y, self.train_Y_emb)
					print ("Cost after epoch %i: loss:%.3f acc: %.3f auc: %.3f" % (epoch+1, epoch_cost,train_acc,train_auroc))
					sys.stdout.flush()
					if epoch>20 and train_acc>0.99:
						break
					if self.save_model is not None and self.use_pretrain is None:
						saver.save(self.sess, self.save_model+str(epoch))
