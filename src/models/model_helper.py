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

def get_data_masks(content, max_len):
	if len(content) > max_len:
		data = content[:max_len]
		data_mask = np.ones(max_len)
	else:
		data = np.concatenate((content, np.zeros(max_len-len(content))), axis=0)
		data_mask = np.concatenate((np.ones(len(content)), np.zeros(max_len-len(content))), axis=0)
	return data, data_mask

def generate_data(posts, ques_list, ans_list, args):
	data_size = len(posts)
	data_posts = np.zeros((data_size, args.post_max_len), dtype=np.int32)
	data_post_masks = np.zeros((data_size, args.post_max_len), dtype=np.float32)
	
	N = args.no_of_candidates	
	data_ques_list = np.zeros((data_size, N, args.ques_max_len), dtype=np.int32)
	data_ques_masks_list = np.zeros((data_size, N, args.ques_max_len), dtype=np.float32)

	data_ans_list = np.zeros((data_size, N, args.ans_max_len), dtype=np.int32)
	data_ans_masks_list = np.zeros((data_size, N, args.ans_max_len), dtype=np.float32)

	for i in range(data_size):
		data_posts[i], data_post_masks[i] = get_data_masks(posts[i], args.post_max_len)
		for j in range(N):
			data_ques_list[i][j], data_ques_masks_list[i][j] = get_data_masks(ques_list[i][j][0], args.ques_max_len)
			data_ans_list[i][j], data_ans_masks_list[i][j] = get_data_masks(ans_list[i][j][0], args.ans_max_len)

	return [data_posts, data_post_masks, data_ques_list, data_ques_masks_list, data_ans_list, data_ans_masks_list]	

def iterate_minibatches(posts, post_masks, ques_list, ques_masks_list, ans_list, ans_masks_list, post_ids, batch_size, shuffle=False):
	if shuffle:
		indices = np.arange(posts.shape[0])
		np.random.shuffle(indices)
	for start_idx in range(0, posts.shape[0] - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield posts[excerpt], post_masks[excerpt], ques_list[excerpt], ques_masks_list[excerpt], ans_list[excerpt], ans_masks_list[excerpt], post_ids[excerpt]

def get_rank(preds, labels):
	preds = np.array(preds)
	correct = np.where(labels==1)[0][0]
	sort_index_preds = np.argsort(preds)
	desc_sort_index_preds = sort_index_preds[::-1] #since ascending sort and we want descending
	rank = np.where(desc_sort_index_preds==correct)[0][0]
	return rank+1

def shuffle(q, qm, a, am, l, r):
	shuffled_q = np.zeros(q.shape, dtype=np.int32)
	shuffled_qm = np.zeros(qm.shape, dtype=np.float32)
	shuffled_a = np.zeros(a.shape, dtype=np.int32)
	shuffled_am = np.zeros(am.shape, dtype=np.float32)
	shuffled_l = np.zeros(l.shape, dtype=np.int32)
	shuffled_r = np.zeros(r.shape, dtype=np.int32)
	
	for i in range(len(q)):
		indexes = range(len(q[i]))
		random.shuffle(indexes)
		for j, index in enumerate(indexes):
			shuffled_q[i][j] = q[i][index]
			shuffled_qm[i][j] = qm[i][index]
			shuffled_a[i][j] = a[i][index]
			shuffled_am[i][j] = am[i][index]
			shuffled_l[i][j] = l[i][index]
			shuffled_r[i][j] = r[i][index]
			
	return shuffled_q, shuffled_qm, shuffled_a, shuffled_am, shuffled_l, shuffled_r


def write_test_predictions(out_file, postId, utilities, ranks):
	lstring = "[%s]: " % (postId)
	N = len(utilities)
	scores = [0]*N
	for i in range(N):
		scores[ranks[i]] = utilities[i]
	for i in range(N):
		lstring += "%f " % (scores[i])
	out_file_o = open(out_file, 'a')
	out_file_o.write(lstring + '\n')
	out_file_o.close()

def get_annotations(line):
	set_info, post_id, best, valids, confidence = line.split(',')
	sitename = set_info.split('_')[1]
	best = [int(best)]
	valids = [int(v) for v in valids.split()]
	confidence = int(confidence)
	return post_id, sitename, best, valids, confidence

def evaluate_using_human_annotations(args, preds):
	human_annotations_file = open(args.test_human_annotations, 'r')
	best_acc_on10 = 0
	best_acc_on9 = 0

	valid_acc_on10 = 0
	valid_acc_on9 = 0

	total = 0
	total_best_on9 = 0	

	for line in human_annotations_file.readlines():
		line = line.strip('\n')
		splits = line.split('\t')
		post_id, sitename, best, valids, confidence = get_annotations(splits[0])
		if len(splits) > 1:
			post_id2, sitename2, best2, valids2, confidence2 = get_annotations(splits[1])		
			assert(sitename == sitename2)
			assert(post_id == post_id2)
			best += [best2]
			valids += valids2

		if best != 0:
			total_best_on9 += 1

		post_id = sitename+'_'+post_id
		pred = preds[post_id].index(max(preds[post_id]))

		if pred in best:
			best_acc_on10 += 1
			if best != 0:
				best_acc_on9 +=1

		if pred in valids:
			valid_acc_on10 += 1
			if pred != 0:
				valid_acc_on9 += 1

		total += 1

	print 
	print '\t\tBest acc on 10: %.2f  Valid acc on 10: %.2f' % (best_acc_on10*100.0/total, valid_acc_on10*100.0/total)
	print '\t\tBest acc on 9:  %.2f  Valid acc on 9:  %.2f' % (best_acc_on9*100.0/total_best_on9, valid_acc_on9*100.0/total)

