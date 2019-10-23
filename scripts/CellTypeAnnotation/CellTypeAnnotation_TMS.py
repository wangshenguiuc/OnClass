import sys
from OnClass.utils import *
from OnClass.OnClassPred import OnClassPred

## read TMS FACS data
DATA_DIR = '../../../OnClass_data/'
data_file = DATA_DIR + 'raw_data/tabula-muris-senis-facs'
X, Y = read_data(filename=data_file, DATA_DIR = DATA_DIR)

## split TMS into training and test
train_X, train_Y_str, test_X, test_Y_str = SplitTrainTest(X, Y)

## embedd the cell ontology
unseen_l, l2i, i2l, onto_net, Y_emb, cls2cls = ParseCLOnto(train_Y_str, DATA_DIR = DATA_DIR)
train_Y = MapLabel2CL(train_Y_str, l2i)
test_Y = MapLabel2CL(test_Y_str, l2i)
unseen_l = MapLabel2CL(unseen_l, l2i)

print ('number of training cells:%d\nnumber of test cells:%d\nnumber of classes in cell ontology:%d\nnumber of unseen classes in test:%d\nnumber of seen classes:%d' % (len(train_Y), len(test_Y), len(l2i), len(set(test_Y) - set(train_Y)),len(set(train_Y))))

## train and predict
OnClass_obj = OnClassPred()
OnClass_obj.train(train_X, train_Y, Y_emb)
test_Y_pred = OnClass_obj.predict(test_X)

## evaluate
res = evaluate(test_Y_pred, train_Y, test_Y, unseen_l, combine_unseen = False)
for metric in res:
	print ('%s\t%.3f' % (metric, res[metric]))
