import argparse
import csv
import sys
import cPickle as p

def read_data(post_data_tsv, qa_data_tsv):
	posts = {}
	titles = {}
	ques_lists = {}
	ans_lists = {}
	with open(post_data_tsv, 'rb') as tsvfile:
		tsv_reader = csv.reader(tsvfile, delimiter='\t')
		for row in tsv_reader:
			post_id = row['postid']
			titles[post_id] = row['title']
			posts[post_id] = row['post']
	with open(qa_data_tsv, 'rb') as tsvfile:
		tsv_reader = csv.reader(tsvfile, delimiter='\t')
		for row in tsv_reader:
			post_id = row['postid']
			ques_lists[post_id] = [row['q1'], row['q2'], row['q3'], row['q4'], row['q5'], row['q6'], row['q7'], row['q8'], row['q9'], row['q10']]
			ans_lists[post_id] = [row['a1'], row['a2'], row['a3'], row['a4'], row['a5'], row['a6'], row['a7'], row['a8'], row['a9'], row['a10']]
	return posts, titles, ques_lists, ans_lists

def read_ids(ids_file):	
	ids = [curr_id.strip('\n') for curr_id in open(ids_file, 'r').readlines().split()]
	return ids

def generate_neural_vectors(posts, titles, ques_lists, ans_lists, post_ids, vocab, N, split):
	post_vectors = []
	ques_list_vectors = []
	ans_list_vectors = []
	for post_id in post_ids:
		post_vectors.append(get_indices(titles[post_id] + ' ' + posts[post_id], vocab))
		ques_list_vector = [None]*N
		ans_list_vector = [None]*N
		for k in range(N):
			ques_list_vector[k] = get_indices(ques_lists[post_id][k], vocab)
			ans_list_vector[k] = get_indices(ans_lists[post_id][k], vocab)
		ques_list_vectors.append(ques_list_vector)
		ans_list_vectors.append(ans_list_vector)
	dirname = os.path.dirname(args.train_ids)
	p.dump(post_ids, open(os.path.join(dirname, 'post_ids_'+split+'.p'), 'wb'))
	p.dump(post_vectors, open(os.path.join(dirname, 'post_vectors_'+split+'.p'), 'wb'))
	p.dump(ques_list_vectors, open(os.path.join(dirname, 'ques_list_vectors_'+split+'.p'), 'wb'))
	p.dump(ans_list_vectors, open(os.path.join(dirname, 'ans_list_vectors_'+split+'.p'), 'wb'))
	
def main(args):
	vocab = p.load(open(args.vocab, 'rb'))
	train_ids = read_ids(args.train_ids)
	tune_ids = read_ids(args.tune_ids)
	test_ids = read_ids(args.test_ids)
	N = args.no_of_candidates
	posts, titles, ques_lists, ans_lists = read_data(args.post_data_tsv, args.qa_data_tsv)
	generate_neural_vectors(posts, titles, ques_lists, ans_lists, train_ids, vocab, N, 'train')
	generate_neural_vectors(posts, titles, ques_lists, ans_lists, tune_ids, vocab, N, 'tune')
	generate_neural_vectors(posts, titles, ques_lists, ans_lists, test_ids, vocab, N, 'test')

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--post_data_tsv", type = str)
	argparser.add_argument("--qa_data_tsv", type = str)
	argparser.add_argument("--train_ids", type = str)
	argparser.add_argument("--tune_ids", type = str)
	argparser.add_argument("--test_ids", type = str)
	argparser.add_argument("--vocab", type = str)
	argparser.add_argument("--no_of_candidates", type = int, default = 10)
	args = argparser.parse_args()
	print args
	print ""
	main(args)

