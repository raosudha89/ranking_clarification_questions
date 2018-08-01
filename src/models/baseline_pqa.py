import sys
import theano, lasagne
import numpy as np
import cPickle as p
import theano.tensor as T
import pdb
import time
DEPTH = 5
from lstm_helper import *
from model_helper import *

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
	loss = 0.0
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
		loss += T.mean(lasagne.objectives.binary_crossentropy(pqa_preds[i*N+i], labels[:,i]))
	
	squared_errors = [None]*(N*N)
	for i in range(N):
		for j in range(N):
			squared_errors[i*N+j] = lasagne.objectives.squared_error(ans_out[i], ans_out[j])
	post_ques_ans_dense_params = lasagne.layers.get_all_params(l_post_ques_ans_dense, trainable=True)

	all_params = post_lstm_params + ques_lstm_params + ans_lstm_params + post_ques_ans_dense_params
	#print 'Params in concat ', lasagne.layers.count_params(l_post_ques_ans_dense)
	loss += args.rho * sum(T.sum(l ** 2) for l in all_params)

	updates = lasagne.updates.adam(loss, all_params, learning_rate=args.learning_rate)
	
	train_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
									[loss] + pqa_preds + squared_errors, updates=updates)
	test_fn = theano.function([posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, labels], \
									[loss] + pqa_preds + squared_errors,)
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
	batch_size = args.batch_size
	
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
		probs = out[1:1+N*N]
		errors = out[1+N*N:]
		probs = np.transpose(probs, (1, 0, 2))
		probs = probs[:,:,0]
		errors = np.transpose(errors, (1, 0, 2))
		errors = errors[:,:,0]
		cost += loss
		for j in range(batch_size):
			preds = [0.0]*N
			for k in range(N):
				preds[k] = probs[j][k*N+k]
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

	lstring = '%s: epoch:%d, cost:%f, acc:%f, mrr:%f,time:%d' % \
				(fold_name, epoch, cost*1.0/num_batches, corr*1.0/total, mrr*1.0/total, time.time()-start)

	recall = [round(curr_r*1.0/total, 3) for curr_r in recall]	
	recall_str = '['
	for r in recall:
		recall_str += '%.3f ' % r
	recall_str += ']\n'
	
	print lstring
	print recall

def baseline_pqa(word_embeddings, vocab_size, word_emb_dim, freeze, args, train, test):
	start = time.time()
	print 'compiling pqa graph...'
	train_fn, test_fn, = build(word_embeddings, vocab_size, word_emb_dim, args, freeze=freeze)
	print 'done! Time taken: ', time.time()-start

	# train network
	for epoch in range(args.no_of_epochs):
		validate(train_fn, 'TRAIN', epoch, train, args)
		validate(test_fn, '\t TEST', epoch, test, args, args.test_predictions_output)
		print "\n"
