import sys, os
import argparse
from parse import *
from post_ques_ans_generator import * 
from helper import *
import time
import numpy as np
import cPickle as p
import pdb
import random

def get_similar_docs(lucene_similar_docs):
	lucene_similar_docs_file = open(lucene_similar_docs, 'r')
	similar_docs = {}
	for line in lucene_similar_docs_file.readlines():
		parts = line.split()
		if len(parts) > 1:
			similar_docs[parts[0]] = parts[1:]
		else:
			similar_docs[parts[0]] = []
	return similar_docs

def generate_docs_for_lucene(post_ques_answers, posts, output_dir):
	for postId in post_ques_answers:
		f = open(os.path.join(output_dir, str(postId) + '.txt'), 'w')
		content = ' '.join(posts[postId].title).encode('utf-8') + ' ' + ' '.join(posts[postId].body).encode('utf-8')
		f.write(content)
		f.close()
		
def create_tsv_files(post_data_tsv, qa_data_tsv, post_ques_answers, lucene_similar_posts):
	lucene_similar_posts = get_similar_docs(lucene_similar_posts)
	similar_posts = {}
	for line in lucene_similar_posts.readlines():
		splits = line.strip('\n').split()
		if len(splits) < 11:
			continue
		postId = splits[0]
		similar_posts[postId] = splits[1:11]
	post_data_tsv_file = open(post_data_tsv, 'w')
	post_data_tsv_file.write('postid\ttitle\tpost\n')
	qa_data_tsv_file = open(qa_data_tsv, 'w')
	qa_data_tsv_file.write('postid\tq1\tq2\tq3\tq4\tq5\tq6\tq7\tq8\tq9\tq10\ta1\ta2\ta3\ta4\ta5\ta6\ta7\ta8\ta9\ta10\n')
	for postId in similar_posts:
		post_data_tsv_file.write('%s\t%s\t%s\n' % (postId, \
													' '.join(post_ques_answers[postId].post_title), \
													' '.join(post_ques_answers[postId].post)))
		line = postId
		for i in range(10):
			line += '\t%s' % ' '.join(post_ques_answers[similar_posts[postId][i]].question_comment)
		for i in range(10):
			line += '\t%s' % ' '.join(post_ques_answers[similar_posts[postId][i]].answer)
		line += '\n'
		qa_data_tsv_file.write(line)

def main(args):
	start_time = time.time()
	print 'Parsing posts...'
	post_parser = PostParser(args.posts_xml)
	post_parser.parse()
	posts = post_parser.get_posts()
	print 'Size: ', len(posts)
	print 'Done! Time taken ', time.time() - start_time
	print

	start_time = time.time()
	print 'Parsing posthistories...'
	posthistory_parser = PostHistoryParser(args.posthistory_xml)
	posthistory_parser.parse()
	posthistories = posthistory_parser.get_posthistories()
	print 'Size: ', len(posthistories)
	print 'Done! Time taken ', time.time() - start_time
	print

	start_time = time.time()
	print 'Parsing question comments...'
	comment_parser = CommentParser(args.comments_xml)
	comment_parser.parse_all_comments()
	question_comments = comment_parser.get_question_comments()
	all_comments = comment_parser.get_all_comments()
	print 'Size: ', len(question_comments)
	print 'Done! Time taken ', time.time() - start_time
	print

	start_time = time.time()
	print 'Loading vocab'	
	vocab = p.load(open(args.vocab, 'rb'))
	print 'Done! Time taken ', time.time() - start_time
	print
	
	start_time = time.time()
	print 'Loading word_embeddings'	
	word_embeddings = p.load(open(args.word_embeddings, 'rb'))
	word_embeddings = np.asarray(word_embeddings, dtype=np.float32)
	print 'Done! Time taken ', time.time() - start_time
	print

	start_time = time.time()
	print 'Generating post_ques_ans...'
	post_ques_ans_generator = PostQuesAnsGenerator()
	post_ques_answers = post_ques_ans_generator.generate(posts, question_comments, all_comments, posthistories, vocab, word_embeddings)
	print 'Size: ', len(post_ques_answers)
	print 'Done! Time taken ', time.time() - start_time
	print
	
	generate_docs_for_lucene(post_ques_answers, posts, args.lucene_docs_dir)
	os.system('cd %s && sh run_lucene.sh %s' % (args.lucene_dir, os.path.dirname(args.post_data_tsv)))

	create_tsv_files(args.post_data_tsv, args.qa_data_tsv, post_ques_answers, args.lucene_similar_posts)

if __name__ == "__main__":
	argparser = argparse.ArgumentParser(sys.argv[0])
	argparser.add_argument("--posts_xml", type = str)
	argparser.add_argument("--comments_xml", type = str)
	argparser.add_argument("--posthistory_xml", type = str)
	argparser.add_argument("--lucene_dir", type = str)
	argparser.add_argument("--lucene_docs_dir", type = str)
	argparser.add_argument("--lucene_similar_posts", type = str)
	argparser.add_argument("--word_embeddings", type = str)
	argparser.add_argument("--vocab", type = str)
	argparser.add_argument("--no_of_candidates", type = int, default = 10)
	argparser.add_argument("--site_name", type = str)
	argparser.add_argument("--post_data_tsv", type = str)
	argparser.add_argument("--qa_data_tsv", type = str)
	args = argparser.parse_args()
	print args
	print ""
	main(args)

