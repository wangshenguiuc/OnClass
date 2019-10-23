import sys
import numpy as np
import os
from utils import *

from OnClass import OnClass

#repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'
#sys.path.append(repo_dir)
#os.chdir(repo_dir)


data_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/SingleCell/tabula_muris_senis/NewAnnotation_20190912/'
data_file = data_dir+'tabula-muris-senis-facs-reannotated-except-for-marrow.h5ad'

## read data
X, Y = read_data(filename=data_file)

## split training and test
train_X, train_Y_str, test_X, test_Y_str = SplitTrainTest(X, Y)

## embedd the cell ontology
unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str)
train_Y = MapLabel2CL(train_Y_str, l2i)
test_Y = MapLabel2CL(test_Y_str, l2i)

print ('number of training cells:%d\nnumber of test cells:%d\nnumber of classes in cell ontology:%d\nnumber of unseen classes in test:%d\nnumber of seen classes:%d' % (len(train_Y), len(test_Y), len(l2i), len(set(test_Y) - set(train_Y)),len(set(train_Y))))

## train and predict
OnClass_obj = OnClass()
OnClass_obj.train(train_X, train_Y, Y_emb)
test_Y_pred = OnClass_obj.predict(test_X)

## evaluate
res = evaluate(test_Y_pred, train_Y, test_Y, unseen_l, combine_unseen = False)
for metric in res:
	print ('%s\t%.3f' % (res, res[metric]))
