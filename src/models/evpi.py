import sys, os
import argparse
import theano, lasagne
import numpy as np
import cPickle as p
import theano.tensor as T
from collections import Counter
import pdb
import time
import random, math
DEPTH = 3
DEPTH_A = 2
from lstm_helper import *
from model_helper import *

def cos_sim_fn(v1, v2):
	numerator = T.sum(v1*v2, axis=1)
	denominator = T.sqrt(T.sum(v1**2, axis=1) * T.sum(v2**2, axis=1))
	val = numerator/denominator
	return T.gt(val,0) * val + T.le(val,0) * 0.001

def custom_sim_fn(v1, v2):
	val = cos_sim_fn(v1, v2)
	val = val - 0.95
	return T.gt(val,0) * T.exp(val)

def answer_model(post_out, ques_out, ques_emb_out, ans_out, ans_emb_out, labels, args):
	# Pr(a|p,q)
	N = args.no_of_candidates
	post_ques = T.concatenate([post_out, ques_out[0]], axis=1)
	hidden_dim = 200
	l_post_ques_in = lasagne.layers.InputLayer(shape=(args.batch_size, 2*args.hidden_dim), input_var=post_ques)
	l_post_ques_denses = [None]*DEPTH_A
	for k in range(DEPTH_A):
		if k == 0:
			l_post_ques_denses[k] = lasagne.layers.DenseLayer(l_post_ques_in, num_units=hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
		else:
			l_post_ques_denses[k] = lasagne.layers.DenseLayer(l_post_ques_denses[k-1], num_units=hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
	
	l_post_ques_dense = lasagne.layers.DenseLayer(l_post_ques_denses[-1], num_units=1,\
												nonlinearity=lasagne.nonlinearities.sigmoid)
	post_ques_dense_params = lasagne.layers.get_all_params(l_post_ques_denses, trainable=True)		
	#print 'Params in post_ques ', lasagne.layers.count_params(l_post_ques_denses)
	
	for i in range(1, N):
		post_ques = T.concatenate([post_out, ques_out[i]], axis=1)
		l_post_ques_in_ = lasagne.layers.InputLayer(shape=(args.batch_size, 2*args.hidden_dim), input_var=post_ques)
		for k in range(DEPTH_A):
			if k == 0:
				l_post_ques_dense_ = lasagne.layers.DenseLayer(l_post_ques_in_, num_units=hidden_dim,\
																nonlinearity=lasagne.nonlinearities.rectify,\
																W=l_post_ques_denses[k].W,\
																b=l_post_ques_denses[k].b)
			else:
				l_post_ques_dense_ = lasagne.layers.DenseLayer(l_post_ques_dense_, num_units=hidden_dim,\
																nonlinearity=lasagne.nonlinearities.rectify,\
																W=l_post_ques_denses[k].W,\
																b=l_post_ques_denses[k].b)
		l_post_ques_dense_ = lasagne.layers.DenseLayer(l_post_ques_dense_, num_units=1,\
													  nonlinearity=lasagne.nonlinearities.sigmoid)
	
	ques_sim = [None]*(N*N)
	pq_a_squared_errors = [None]*(N*N)
	for i in range(N):
		for j in range(N):
			ques_sim[i*N+j] = custom_sim_fn(ques_emb_out[i], ques_emb_out[j])
			pq_a_squared_errors[i*N+j] = 1-cos_sim_fn(ques_emb_out[i], ans_emb_out[j])
	
	pq_a_loss = 0.0	
	for i in range(N):
		pq_a_loss += T.mean(labels[:,i] * pq_a_squared_errors[i*N+i])
		for j in range(N):
			if i == j:
				continue
			pq_a_loss += T.mean(labels[:,i] * pq_a_squared_errors[i*N+j] * ques_sim[i*N+j])

	return ques_sim, pq_a_squared_errors, pq_a_loss, post_ques_dense_params

def utility_calculator(post_out, ques_out, ques_emb_out, ans_out, ques_sim, pq_a_squared_errors, labels, args):
	# U(p+a)
	N = args.no_of_candidates
	pqa_loss = 0.0
	pqa_preds = [None]*(N*N)
	post_ques_ans = T.concatenate([post_out, ques_out[0], ans_out[0]], axis=1)
	l_post_ques_ans_in = lasagne.layers.InputLayer(shape=(args.batch_size, 3*args.hidden_dim), input_var=post_ques_ans)
	l_post_ques_ans_denses = [None]*DEPTH
	for k in range(DEPTH):
		if k == 0:
			l_post_ques_ans_denses[k] = lasagne.layers.DenseLayer(l_post_ques_ans_in, num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
		else:
			l_post_ques_ans_denses[k] = lasagne.layers.DenseLayer(l_post_ques_ans_denses[k-1], num_units=args.hidden_dim,\
															nonlinearity=lasagne.nonlinearities.rectify)
	l_post_ques_ans_dense = lasagne.layers.DenseLayer(l_post_ques_ans_denses[-1], num_units=1,\
												nonlinearity=lasagne.nonlinearities.sigmoid)
	pqa_preds[0] = lasagne.layers.get_output(l_post_ques_ans_dense)

	pqa_loss += T.mean(lasagne.objectives.binary_crossentropy(pqa_preds[0], labels[:,0]))

	for i in range(N):
		for j in range(N):
			if i == 0 and j == 0:
				continue
			post_ques_ans = T.concatenate([post_out, ques_out[i], ans_out[j]], axis=1)
			l_post_ques_ans_in_ = lasagne.layers.InputLayer(shape=(args.batch_size, 3*args.hidden_dim), input_var=post_ques_ans)
			for k in range(DEPTH):
				if k == 0:
					l_post_ques_ans_dense_ = lasagne.layers.DenseLayer(l_post_ques_ans_in_, num_units=args.hidden_dim,\
																nonlinearity=lasagne.nonlinearities.rectify,\
																W=l_post_ques_ans_denses[k].W,\
																b=l_post_ques_ans_denses[k].b)
				else:
					l_post_ques_ans_dense_ = lasagne.layers.DenseLayer(l_post_ques_ans_dense_, num_units=args.hidden_dim,\
																nonlinearity=lasagne.nonlinearities.rectify,\
																W=l_post_ques_ans_denses[k].W,\
																b=l_post_ques_ans_denses[k].b)
			l_post_ques_ans_dense_ = lasagne.layers.DenseLayer(l_post_ques_ans_dense_, num_units=1,\
													   nonlinearity=lasagne.nonlinearities.sigmoid)
			pqa_preds[i*N+j] = lasagne.layers.get_output(l_post_ques_ans_dense_)
		pqa_loss += T.mean(lasagne.objectives.binary_crossentropy(pqa_preds[i*N+i], labels[:,i]))
	post_ques_ans_dense_params = lasagne.layers.get_all_params(l_post_ques_ans_dense, trainable=True)
	#print 'Params in post_ques_ans ', lasagne.layers.count_params(l_post_ques_ans_dense)

	return pqa_loss, post_ques_ans_dense_params, pqa_preds

def build(word_embeddings, len_voc, word_emb_dim, args, freeze=False):
	# input theano vars
	posts = T.imatrix()
	post_masks = T.fmatrix()
	ques_list = T.itensor3()
	ques_masks_list = T.ftensor3()
	ans_list = T.itensor3()
	ans_masks_list = T.ftensor3()
	labels = T.imatrix()
	N = args.no_of_candidates

	post_out, post_lstm_params = build_lstm(posts, post_masks, args.post_max_len, \
												  word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)	
	ques_out, ques_emb_out, ques_lstm_params = build_list_lstm(ques_list, ques_masks_list, N, args.ques_max_len, \
											word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)
	ans_out, ans_emb_out, ans_lstm_params = build_list_lstm(ans_list, ans_masks_list, N, args.ans_max_len, \
											word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)

	ques_sim, pq_a_squared_errors, pq_a_loss, post_ques_dense_params \
											 = answer_model(post_out, ques_out, ques_emb_out, ans_out, ans_emb_out, labels, args)	

	all_params = post_lstm_params + ques_lstm_params + post_ques_dense_params
	
	post_out, post_lstm_params = build_lstm(posts, post_masks, args.post_max_len, \
												  word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)	
	ques_out, ques_emb_out, ques_lstm_params = build_list_lstm(ques_list, ques_masks_list, N, args.ques_max_len, \
											word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)
	ans_out, ans_emb_out, ans_lstm_params = build_list_lstm(ans_list, ans_masks_list, N, args.ans_max_len, \
											word_embeddings, word_emb_dim, args.hidden_dim, len_voc, args.batch_size)

	pqa_loss, post_ques_ans_dense_params, pqa_preds = utility_calculator(post_out, ques_out, ques_emb_out, ans_out, \
																			ques_sim, pq_a_squared_errors, labels, args)	

	all_params += post_lstm_params + ques_lstm_params + ans_lstm_params
	all_params += post_ques_ans_dense_params

	loss = pq_a_loss + pqa_loss	
	loss += args.rho * sum(T.sum(l ** 2) for l in all_params)

	updates = lasagne.updates.adam(loss, all_params, learning_rate=args.learning_rate)
	
	train_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
									[loss, pq_a_loss, pqa_loss] + pq_a_squared_errors + ques_sim + pqa_preds, updates=updates)
	test_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
									[loss, pq_a_loss, pqa_loss] + pq_a_squared_errors + ques_sim + pqa_preds,)
	return train_fn, test_fn

def validate(val_fn, fold_name, epoch, fold, args, out_file=None):
	start = time.time()
	num_batches = 0
	cost = 0
	pq_a_cost = 0
	utility_cost = 0
	corr = 0
	mrr = 0
	total = 0
	N = args.no_of_candidates
	recall = [0]*N
	
	if out_file:
		out_file_o = open(out_file+'.epoch%d' % epoch, 'w')
		out_file_o.close()
	posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, post_ids = fold
	labels = np.zeros((len(post_ids), N), dtype=np.int32)
	ranks = np.zeros((len(post_ids), N), dtype=np.int32)
	labels[:,0] = 1
	for j in range(N):
		ranks[:,j] = j
	ques_list, ques_masks_list, ans_list, ans_masks_list, labels, ranks = shuffle(ques_list, ques_masks_list, \
																					ans_list, ans_masks_list, labels, ranks)
	for p, pm, q, qm, a, am, l, r, ids in iterate_minibatches(posts, post_masks, ques_list, ques_masks_list, \
																ans_list, ans_masks_list, labels, ranks, \
														 		post_ids, args.batch_size, shuffle=False):
		q = np.transpose(q, (1, 0, 2))
		qm = np.transpose(qm, (1, 0, 2))
		a = np.transpose(a, (1, 0, 2))
		am = np.transpose(am, (1, 0, 2))
		
		out = val_fn(p, pm, q, qm, a, am, l)
		loss = out[0]
		pq_a_loss = out[1]
		pqa_loss = out[2]
	
		pq_a_errors = out[3: 3+(N*N)]
		pq_a_errors = np.transpose(pq_a_errors)
		
		ques_sim = out[3+(N*N): 3+(N*N)+(N*N)]
		ques_sim = np.transpose(ques_sim)
	
		pqa_preds = out[3+(N*N)+(N*N):]
		pqa_preds = np.array(pqa_preds)[:,:,0]
		pqa_preds = np.transpose(pqa_preds)
			
		cost += loss
		pq_a_cost += pq_a_loss
		utility_cost += pqa_loss
		
		for j in range(args.batch_size):
			preds = [0.0]*N
			for k in range(N):
				preds[k] = pqa_preds[j][k*N+k]
				for m in range(N):
					if m == k:
						continue
					preds[k] += 0.1*math.exp(pq_a_errors[j][k*N+m])*ques_sim[j][k*N+m] * pqa_preds[j][k*N+m]
			rank = get_rank(preds, l[j])
			if rank == 1:
				corr += 1
			mrr += 1.0/rank
			for index in range(N):
				if rank <= index+1:
					recall[index] += 1
			total += 1
			if out_file:
				write_test_predictions(out_file, ids[j], preds, r[j], epoch)
		num_batches += 1
	
	lstring = '%s: epoch:%d, cost:%f, pq_a_cost:%f, utility_cost:%f, acc:%f, mrr:%f,time:%d\n' % \
				(fold_name, epoch, cost*1.0/num_batches, pq_a_cost*1.0/num_batches, utility_cost*1.0/num_batches, \
					corr*1.0/total, mrr*1.0/total, time.time()-start)
	recall = [round(curr_r*1.0/total, 3) for curr_r in recall]	
	recall_str = '['
	for r in recall:
		recall_str += '%.3f ' % r
	recall_str += ']\n'
	
	outfile = open(args.stdout_file, 'a')
	outfile.write(lstring+'\n')
	outfile.write(recall_str+'\n')
	outfile.close()

	print lstring
	print recall

def evpi(word_embeddings, vocab_size, word_emb_dim, freeze, args, train, test):
	outfile = open(args.stdout_file, 'w')
	outfile.close()
	
	print 'Compiling graph...'
	start = time.time()
	train_fn, test_fn = build(word_embeddings, vocab_size, word_emb_dim, args, freeze=freeze)
	print 'done! Time taken: ', time.time() - start

	# train network
	for epoch in range(args.no_of_epochs):
		validate(train_fn, 'TRAIN', epoch, train, args)
		validate(test_fn, '\t TEST', epoch, test, args, args.test_predictions_output)
		print "\n"

