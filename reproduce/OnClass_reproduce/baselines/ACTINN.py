import numpy as np
import pandas as pd
import sys
import math
import collections
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import argparse
import timeit
tf.logging.set_verbosity(tf.logging.ERROR)

run_time = timeit.default_timer()
from tensorflow.python.framework import ops

def get_parser(parser=None):
	if parser == None:
		parser = argparse.ArgumentParser()
	parser.add_argument("-trs", "--train_set", type=str, help="Training set file path.")
	parser.add_argument("-trl", "--train_label", type=str, help="Training label file path.")
	parser.add_argument("-ts", "--test_set", type=str, help="Training set file path.")
	parser.add_argument("-lr", "--learning_rate", type=float, help="Learning rate (default: 0.0001)", default=0.0001)
	parser.add_argument("-ne", "--num_epochs", type=int, help="Number of epochs (default: 50)", default=50)
	parser.add_argument("-ms", "--minibatch_size", type=int, help="Minibatch size (default: 128)", default=128)
	parser.add_argument("-pc", "--print_cost", type=bool, help="Print cost when training (default: True)", default=True)
	parser.add_argument("-op", "--output_probability", type=bool, help="Output the probabilities for each cell being the cell types in the training data (default: False)", default=False)
	return parser



# Get common genes, normalize  and scale the sets
def scale_sets(sets):
	# input -- a list of all the sets to be scaled
	# output -- scaled sets

	common_genes = set(sets[0].index)
	for i in range(1, len(sets)):
		common_genes = set.intersection(set(sets[i].index),common_genes)
	common_genes = sorted(list(common_genes))
	sep_point = [0]
	for i in range(len(sets)):
		sets[i] = sets[i].loc[common_genes,]
		sep_point.append(sets[i].shape[1])

	#sets = [sets[0], sets[1]]
	total_set = pd.concat(sets, axis=1, sort=False)
	#print (total_set)
	#print ('actinn d0',np.shape(total_set))
	total_set = total_set.loc[total_set.sum(axis=1)>0, :]
	#print (total_set)
	#print ('actinn d1',np.shape(total_set))
	total_set = np.array(total_set, dtype=np.float32)
	total_set = np.divide(total_set, np.sum(total_set, axis=0, keepdims=True)) * 10000
	total_set = np.log2(total_set+1)
	expr = np.sum(total_set, axis=1)
	total_set = total_set[np.logical_and(expr >= np.percentile(expr, 1), expr <= np.percentile(expr, 99)),]
	#print (total_set)
	#print ('actinn d2',np.shape(total_set))
	cv = np.std(total_set, axis=1) / np.mean(total_set, axis=1)
	total_set = total_set[np.logical_and(cv >= np.percentile(cv, 1), cv <= np.percentile(cv, 99)),]
	#print (total_set)
	#print ('actinn d4',np.shape(total_set))
	#print (total_set.T, np.shape(total_set.T))
	for i in range(len(sets)):
		sets[i] = total_set[:, sum(sep_point[:(i+1)]):sum(sep_point[:(i+2)])]

	return sets


# Get common genes, normalize  and scale the sets
def my_scale_sets(sets):
	sep_point = [0]
	for i in range(len(sets)):
		sep_point.append(sets[i].shape[1])
	total_set = np.column_stack([sets[0], sets[1]])
	#total_set = total_set.loc[total_set.sum(axis=1)>0, :]
	#total_set = total_set[np.where(np.sum(total_set,axis=1)>0)[0], :]
	total_set = np.array(total_set, dtype=np.float32)
	print (np.shape(total_set))

	print ('d1')
	#total_set = np.divide(total_set, np.sum(total_set, axis=0, keepdims=True)) * 10000
	total_set = np.log1p(total_set)
	'''
	expr = np.sum(total_set, axis=1)
	total_set = total_set[np.logical_and(expr >= np.percentile(expr, 1), expr <= np.percentile(expr, 99)),]
	print (np.shape(total_set))
	print ('d2')

	cv = np.std(total_set, axis=1) / np.mean(total_set, axis=1)
	total_set = total_set[np.logical_and(cv >= np.percentile(cv, 1), cv <= np.percentile(cv, 99)),]
	print (np.shape(total_set))
	print ('d3')
	'''
	for i in range(len(sets)):
		sets[i] = total_set[:, sum(sep_point[:(i+1)]):sum(sep_point[:(i+2)])]
	return sets

# Turn labels into matrix
def one_hot_matrix(labels, C):
	# input -- labels (true labels of the sets), C (# types)
	# output -- one hot matrix with shape (# types, # samples)
	C = tf.constant(C, name = "C", dtype=tf.int64)
	labels = np.array(labels)
	labels = tf.cast(labels, "int32")
	C = tf.cast(C, "int32")
	one_hot_matrix = tf.one_hot(labels, C, axis = 0)
	sess = tf.Session()
	one_hot = sess.run(one_hot_matrix)
	sess.close()
	return one_hot



# Make types to labels dictionary
def type_to_label_dict_func(types):
	# input -- types
	# output -- type_to_label dictionary
	type_to_label_dict = {}
	all_type = list(set(types))
	for i in range(len(all_type)):
		type_to_label_dict[all_type[i]] = i
	return type_to_label_dict



# Convert types to labels
def convert_type_to_label(types, type_to_label_dict):
	# input -- list of types, and type_to_label dictionary
	# output -- list of labels
	types = list(types)
	labels = list()
	for type in types:
		labels.append(type_to_label_dict[type])
	return labels



# Function to create placeholders
def create_placeholders(n_x, n_y):
	X = tf.placeholder(tf.float32, shape = (n_x, None))
	Y = tf.placeholder(tf.float32, shape = (n_y, None))
	return X, Y



# Initialize parameters
def initialize_parameters(nf, ln1, ln2, ln3, nt):
	# input -- nf (# of features), ln1 (# nodes in layer1), ln2 (# nodes in layer2), nt (# types)
	# output -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
	tf.set_random_seed(3) # set seed to make the results consistant
	W1 = tf.get_variable("W1", [ln1, nf], initializer = tf.contrib.layers.xavier_initializer(seed = 3))
	b1 = tf.get_variable("b1", [ln1, 1], initializer = tf.zeros_initializer())
	W2 = tf.get_variable("W2", [ln2, ln1], initializer = tf.contrib.layers.xavier_initializer(seed = 3))
	b2 = tf.get_variable("b2", [ln2, 1], initializer = tf.zeros_initializer())
	W3 = tf.get_variable("W3", [ln3, ln2], initializer = tf.contrib.layers.xavier_initializer(seed = 3))
	b3 = tf.get_variable("b3", [ln3, 1], initializer = tf.zeros_initializer())
	W4 = tf.get_variable("W4", [nt, ln3], initializer = tf.contrib.layers.xavier_initializer(seed = 3))
	b4 = tf.get_variable("b4", [nt, 1], initializer = tf.zeros_initializer())
	parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4}
	return parameters



# Forward propagation function
def forward_propagation(X, parameters):
	# function model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
	# input -- dataset with shape (# features, # sample), parameters "W1", "b1", "W2", "b2", "W3", "b3"
	# output -- the output of the last linear unit
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	W3 = parameters['W3']
	b3 = parameters['b3']
	W4 = parameters['W4']
	b4 = parameters['b4']
	# forward calculations
	Z1 = tf.add(tf.matmul(W1, X), b1)
	A1 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(W2, A1), b2)
	A2 = tf.nn.relu(Z2)
	Z3 = tf.add(tf.matmul(W3, A2), b3)
	A3 = tf.nn.relu(Z3)
	Z4 = tf.add(tf.matmul(W4, A3), b4)
	return Z4



# Compute cost
def compute_cost(Z4, Y, parameters, lambd=0.01):
	# input -- Z3 (output of forward propagation with shape (# types, # samples)), Y (true labels, same shape as Z3)
	# output -- tensor of teh cost function
	logits = tf.transpose(Z4)
	labels = tf.transpose(Y)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = labels)) + \
	(tf.nn.l2_loss(parameters["W1"]) + tf.nn.l2_loss(parameters["W2"]) + tf.nn.l2_loss(parameters["W3"]) + tf.nn.l2_loss(parameters["W4"])) * lambd
	return cost



# Get the mini batches
def random_mini_batches(X, Y, mini_batch_size=32, seed=1):
	# input -- X (training set), Y (true labels)
	# output -- mini batches
	ns = X.shape[1]
	mini_batches = []
	np.random.seed(seed)
	# shuffle (X, Y)
	permutation = list(np.random.permutation(ns))
	shuffled_X = X[:, permutation]
	shuffled_Y = Y[:, permutation]
	# partition (shuffled_X, shuffled_Y), minus the end case.
	num_complete_minibatches = int(math.floor(ns/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
		mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	# handling the end case (last mini-batch < mini_batch_size)
	if ns % mini_batch_size != 0:
		mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : ns]
		mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : ns]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	return mini_batches



# Forward propagation for prediction
def forward_propagation_for_predict(X, parameters):
	# input -- X (dataset used to make prediction), papameters after training
	# output -- the output of the last linear unit
	W1 = parameters['W1']
	b1 = parameters['b1']
	W2 = parameters['W2']
	b2 = parameters['b2']
	W3 = parameters['W3']
	b3 = parameters['b3']
	W4 = parameters['W4']
	b4 = parameters['b4']
	Z1 = tf.add(tf.matmul(W1, X), b1)
	A1 = tf.nn.relu(Z1)
	Z2 = tf.add(tf.matmul(W2, A1), b2)
	A2 = tf.nn.relu(Z2)
	Z3 = tf.add(tf.matmul(W3, A2), b3)
	A3 = tf.nn.relu(Z3)
	Z4 = tf.add(tf.matmul(W4, A3), b4)
	return Z4



# Predict function
def predict(X, parameters):
	# input -- X (dataset used to make prediction), papameters after training
	# output -- prediction
	W1 = tf.convert_to_tensor(parameters["W1"])
	b1 = tf.convert_to_tensor(parameters["b1"])
	W2 = tf.convert_to_tensor(parameters["W2"])
	b2 = tf.convert_to_tensor(parameters["b2"])
	W3 = tf.convert_to_tensor(parameters["W3"])
	b3 = tf.convert_to_tensor(parameters["b3"])
	W4 = tf.convert_to_tensor(parameters["W4"])
	b4 = tf.convert_to_tensor(parameters["b4"])
	params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4}
	x = tf.placeholder("float")
	z4 = forward_propagation_for_predict(x, params)
	p = tf.argmax(z4)
	sess = tf.Session()
	prediction = sess.run(p, feed_dict = {x: X})
	return prediction



def predict_probability(X, parameters):
	# input -- X (dataset used to make prediction), papameters after training
	# output -- prediction
	W1 = tf.convert_to_tensor(parameters["W1"])
	b1 = tf.convert_to_tensor(parameters["b1"])
	W2 = tf.convert_to_tensor(parameters["W2"])
	b2 = tf.convert_to_tensor(parameters["b2"])
	W3 = tf.convert_to_tensor(parameters["W3"])
	b3 = tf.convert_to_tensor(parameters["b3"])
	W4 = tf.convert_to_tensor(parameters["W4"])
	b4 = tf.convert_to_tensor(parameters["b4"])
	params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4}
	x = tf.placeholder("float")
	z4 = forward_propagation_for_predict(x, params)
	#p = tf.nn.softmax(z4, axis=0)
	p = tf.exp(z4) / tf.reduce_sum(tf.exp(z4), 0)
	sess = tf.Session()
	prediction = sess.run(p, feed_dict = {x: X})
	return prediction



# Build the model
def model(X_train, Y_train, X_test, starting_learning_rate = 0.0001, num_epochs = 50, minibatch_size = 128, print_cost = True):
	# input -- X_train (training set), Y_train(training labels), X_test (test set), Y_test (test labels),
	# output -- trained parameters
	ops.reset_default_graph() # to be able to rerun the model without overwriting tf variables
	tf.set_random_seed(3)
	seed = 3
	(nf, ns) = X_train.shape
	nt = Y_train.shape[0]
	costs = []
	# create placeholders of shape (nf, nt)
	X, Y = create_placeholders(nf, nt)
	# initialize parameters
	parameters = initialize_parameters(nf=nf, ln1=100, ln2=50, ln3=25, nt=nt)
	# forward propagation: build the forward propagation in the tensorflow graph
	Z4 = forward_propagation(X, parameters)
	# cost function: add cost function to tensorflow graph
	cost = compute_cost(Z4, Y, parameters, 0.005)
	# Use learning rate decay
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(starting_learning_rate, global_step, 1000, 0.95, staircase=True)
	# backpropagation: define the tensorflow optimizer, AdamOptimizer is used.
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	trainer = optimizer.minimize(cost, global_step=global_step)
	# initialize all the variables
	init = tf.global_variables_initializer()
	# start the session to compute the tensorflow graph
	with tf.Session() as sess:
		# run the initialization
		sess.run(init)
		# do the training loop
		for epoch in range(num_epochs):
			epoch_cost = 0.
			num_minibatches = int(ns / minibatch_size)
			seed = seed + 1
			minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
			for minibatch in minibatches:
				# select a minibatch
				(minibatch_X, minibatch_Y) = minibatch

				# run the session to execute the "optimizer" and the "cost", the feedict contains a minibatch for (X,Y).
				_ , minibatch_cost = sess.run([trainer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
				epoch_cost += minibatch_cost / num_minibatches
			# print the cost every epoch
			if print_cost == True:
				print ("Cost after epoch %i: %f" % (epoch+1, epoch_cost))
				costs.append(epoch_cost)
		parameters = sess.run(parameters)
		# calculate the correct predictions
		correct_prediction = tf.equal(tf.argmax(Z4), tf.argmax(Y))
		# calculate accuracy on the test set
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		#print ("ACTINN Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
		return parameters


#python /oak/stanford/groups/rbaltman/swang91/Sheng_repo/tools/single_cell/actinn/ACTINN/actinn_predict.py -trs  -trl /oak/stanford/groups/rbaltman/swang91/Sheng_repo/tools/single_cell/actinn/ACTINN/tms/1m_train.labels -ts /oak/stanford/groups/rbaltman/swang91/Sheng_repo/tools/single_cell/actinn/ACTINN/tms/1m_test.h5 -lr 0.0001 -ne 50 -ms 128 -pc True -op False

def run_ACTINN(trainX, trainY, testX, nseen, num_epochs = 50, minibatch_size = 128, learning_rate = 0.0001):
	#trainX = trainX.toarray()
	#testX = testX.toarray()
	assert(np.min(trainX)>=0)
	assert(np.min(testX)>=0)


	ntrain, ngene = np.shape(trainX)
	ntest, ngene = np.shape(testX)
	genes = ['gene'+str(s).zfill(5) for s in range(ngene)]
	train_ind = ['train_ind'+str(s) for s in range(ntrain)]
	test_ind = ['test_ind'+str(s) for s in range(ntest)]
	train_ind = np.array(train_ind)
	genes = np.array(genes)
	#print (trainX.getformat())
	train_set = pd.DataFrame(data = trainX, columns = genes, index = train_ind).T
	test_set = pd.DataFrame(data = testX, columns = genes, index = test_ind).T
	#parser = get_parser()
	#args = parser.parse_args()

	output_probability = True
	print_cost = True
	barcode = list(test_set.columns)
	nt = nseen
	train_set, test_set = scale_sets([train_set, test_set])
	type_to_label_dict = type_to_label_dict_func(trainY)
	#print ('start')
	label_to_type_dict = {v: k for k, v in type_to_label_dict.items()}
	#print("Cell Types in training set:", type_to_label_dict)
	#print("# Trainng cells:", trainX.shape[0])
	train_label = convert_type_to_label(trainY, type_to_label_dict)
	train_label = one_hot_matrix(train_label, nt)
	parameters = model(train_set, train_label, test_set, \
	learning_rate, num_epochs, minibatch_size, print_cost)
	test_predict = predict_probability(test_set, parameters)
	#test_Y_pred_raw = np.zeros((ntest, nseen))
	test_Y_pred_raw = np.zeros((ntest, nseen))
	for i in label_to_type_dict:
		test_Y_pred_raw[:,label_to_type_dict[i]] = test_predict[i,:]
	tf.reset_default_graph()

	return test_Y_pred_raw
