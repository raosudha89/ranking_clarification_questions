import sys
import argparse
import theano, lasagne
import numpy as np
import cPickle as p
import theano.tensor as T
from collections import Counter
import pdb
import time
import random, math
DEPTH = 5
from lstm_helper import *
from model_helper import *

def build(word_embeddings, len_voc, word_emb_dim, args, freeze=False):

	# input theano vars
	posts = T.imatrix()
	post_masks = T.fmatrix()
	ques_list = T.itensor3()
	ques_masks_list = T.ftensor3()
	labels = T.imatrix()
	N = args.no_of_candidates

	post_out, post_lstm_params = build_lstm(posts, post_masks, args.post_max_len, \
												  word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)	
	ques_out, ques_lstm_params = build_list_lstm(ques_list, ques_masks_list, N, args.ques_max_len, \
											word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)
	
	pq_preds = [None]*N
	post_ques = T.concatenate([post_out, ques_out[0]], axis=1)
	l_post_ques_in = lasagne.layers.InputLayer(shape=(args.batch_size, 2*args.hidden_dim), input_var=post_ques)
	l_post_ques_denses = [None]*DEPTH
	for k in range(DEPTH):
		if k == 0:
			l_post_ques_denses[k] = lasagne.layers.DenseLayer(l_post_ques_in, num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
		else:
			l_post_ques_denses[k] = lasagne.layers.DenseLayer(l_post_ques_denses[k-1], num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
	l_post_ques_dense = lasagne.layers.DenseLayer(l_post_ques_denses[-1], num_units=1,\
												nonlinearity=lasagne.nonlinearities.sigmoid)
	pq_preds[0] = lasagne.layers.get_output(l_post_ques_dense)
	loss = T.sum(lasagne.objectives.binary_crossentropy(pq_preds[0], labels[:,0]))
	for i in range(1, N):
		post_ques = T.concatenate([post_out, ques_out[i]], axis=1)
		l_post_ques_in_ = lasagne.layers.InputLayer(shape=(args.batch_size, 2*args.hidden_dim), input_var=post_ques)
		for k in range(DEPTH):
			if k == 0:
				l_post_ques_dense_ = lasagne.layers.DenseLayer(l_post_ques_in_, num_units=args.hidden_dim,\
																nonlinearity=lasagne.nonlinearities.rectify,\
																W=l_post_ques_denses[k].W,\
																b=l_post_ques_denses[k].b)
			else:
				l_post_ques_dense_ = lasagne.layers.DenseLayer(l_post_ques_dense_, num_units=args.hidden_dim,\
																nonlinearity=lasagne.nonlinearities.rectify,\
																W=l_post_ques_denses[k].W,\
																b=l_post_ques_denses[k].b)
		l_post_ques_dense_ = lasagne.layers.DenseLayer(l_post_ques_dense_, num_units=1,\
													  nonlinearity=lasagne.nonlinearities.sigmoid)
		pq_preds[i] = lasagne.layers.get_output(l_post_ques_dense_)
		loss += T.sum(lasagne.objectives.binary_crossentropy(pq_preds[i], labels[:,i]))

	post_ques_dense_params = lasagne.layers.get_all_params(l_post_ques_dense, trainable=True)

	all_params = post_lstm_params + ques_lstm_params + post_ques_dense_params
	print 'Params in concat ', lasagne.layers.count_params(l_post_ques_dense)
	loss += args.rho * sum(T.sum(l ** 2) for l in all_params)

	updates = lasagne.updates.adam(loss, all_params, learning_rate=args.learning_rate)
	
	train_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, labels], \
									[loss] + pq_preds, updates=updates)
	test_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, labels], \
									[loss] + pq_preds,)
	return train_fn, test_fn

def validate(val_fn, fold_name, epoch, fold, args, out_file=None):
	start = time.time()
	num_batches = 0
	cost = 0
	corr = 0
	mrr = 0
	total = 0
	_lambda = 0.5
	N = args.no_of_candidates
	recall = [0]*N
	
	if out_file:
		out_file_o = open(out_file, 'a')
		out_file_o.write("\nEpoch: %d\n" % epoch)
		out_file_o.close()
	posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, post_ids = fold
	for p, pm, q, qm, a, am, ids in iterate_minibatches(posts, post_masks, ques_list, ques_masks_list,	\
														ans_list, ans_masks_list, post_ids, args.batch_size, shuffle=False):
		l = np.zeros((args.batch_size, N), dtype=np.int32)
		r = np.zeros((args.batch_size, N), dtype=np.int32)
		l[:,0] = 1
		for j in range(N):
			r[:,j] = j
		q, qm, a, am, l, r = shuffle(q, qm, a, am, l, r)
		q = np.transpose(q, (1, 0, 2))
		qm = np.transpose(qm, (1, 0, 2))
		
		pq_out = val_fn(p, pm, q, qm, l)
		loss = pq_out[0]
		pq_preds = pq_out[1:]
		pq_preds = np.transpose(pq_preds, (1, 0, 2))
		pq_preds = pq_preds[:,:,0]
		cost += loss
		for j in range(args.batch_size):
			preds = [0.0]*N
			for k in range(N):
				preds[k] = pq_preds[j][k]
			rank = get_rank(preds, l[j])
			if rank == 1:
				corr += 1
			mrr += 1.0/rank
			for index in range(N):
				if rank <= index+1:
					recall[index] += 1
			total += 1
			if out_file:
				write_test_predictions(out_file, ids[j], preds, r[j])
		num_batches += 1

	lstring = '%s: epoch:%d, cost:%f, acc:%f, mrr:%f,time:%d' % \
				(fold_name, epoch, cost*1.0/num_batches, corr*1.0/total, mrr*1.0/total, time.time()-start)

	recall = [round(curr_r*1.0/total, 3) for curr_r in recall]	
	recall_str = '['
	for r in recall:
		recall_str += '%.3f ' % r
	recall_str += ']\n'
	
	print lstring
	print recall

def baseline_pq(word_embeddings, vocab_size, word_emb_dim, freeze, args, train, test):
	start = time.time()
	print 'compiling pq graph...'
	train_fn, test_fn, = build(word_embeddings, vocab_size, word_emb_dim, args, freeze=freeze)
	print 'done! Time taken: ', time.time()-start

	# train network
	for epoch in range(args.no_of_epochs):
		validate(train_fn, 'TRAIN', epoch, train, args)
		validate(test_fn, '\t TEST', epoch, test, args, args.test_predictions_output)
		print "\n"
